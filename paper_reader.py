import requests
import json
import os
import datetime
import time
from typing import TypedDict, List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
# 自动加载 .env 文件
load_dotenv()

from openai import OpenAI
import re

# 定义状态类型
class PPTState(TypedDict):
    content_url: str
    scraped_text: str
    image_urls: List[str]
    text_summary: str
    paper_title: str  # 新增论文标题字段
    png_images: List[str]  # 新增PNG图片字段
    temp_filename: str  # 新增临时文件名字段

# 初始化OpenAI客户端（兼容Qwen API）
client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    # 移除或替换非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    filename = re.sub(r'\s+', ' ', filename).strip()
    # 限制长度
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def extract_paper_title(text_content: str, url: str) -> str:
    """从文本内容或URL中提取论文标题"""
    title = "未知论文"
    
    try:
        # 方法1：从URL中提取（适用于arXiv）
        if 'arxiv.org' in url:
            arxiv_patterns = [
                r'/abs/(\d+\.\d+)',
                r'/html/(\d+\.\d+)',
                r'/(\d+\.\d+)v?\d*'
            ]
            
            for pattern in arxiv_patterns:
                arxiv_match = re.search(pattern, url)
                if arxiv_match:
                    title = f"arXiv_{arxiv_match.group(1)}"
                    print(f"从URL提取到arXiv编号：{arxiv_match.group(1)}")
                    break
        
        # 方法2：从文本内容中提取标题
        if title == "未知论文":
            # 改进的标题提取模式
            title_patterns = [
                r'Title[:\s]*([^\n\r]+)',
                r'标题[:\s]*([^\n\r]+)',
                r'^([A-Z][^.!?\n\r]{10,100})',  # 以大写字母开头的长句子
                r'^([^.!?\n\r]{15,80})',  # 长度适中的第一行
                r'BannerAgency[^.!?\n\r]*',  # 特定关键词匹配
                r'GPT-[^.!?\n\r]*',  # GPT相关论文
                r'Attention[^.!?\n\r]*',  # Attention相关论文
            ]
            
            lines = text_content.split('\n')
            print(f"检查前10行文本内容：")
            for i, line in enumerate(lines[:10]):
                line = line.strip()
                print(f"  行{i+1}: {line[:50]}...")
                
                if len(line) > 10 and len(line) < 200:  # 合理的标题长度
                    for pattern in title_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            potential_title = match.group(0) if pattern.endswith('*') else match.group(1)
                            potential_title = potential_title.strip()
                            if len(potential_title) > 5:  # 确保标题有意义
                                title = potential_title
                                print(f"找到标题：{title}")
                                break
                    if title != "未知论文":
                        break
        
        # 方法3：从HTML标题标签中提取（如果可用）
        if title == "未知论文" and 'arxiv.org' in url:
            try:
                # 尝试获取HTML页面的标题
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    html_title = soup.find('title')
                    if html_title and html_title.text:
                        title_text = html_title.text.strip()
                        if len(title_text) > 5 and len(title_text) < 200:
                            title = title_text
                            print(f"从HTML标题提取：{title}")
            except Exception as e:
                print(f"从HTML标题提取失败：{e}")
        
        # 清理标题
        title = sanitize_filename(title)
        print(f"最终提取的标题：{title}")
        
    except Exception as e:
        print(f"提取标题时出错：{e}")
        title = "未知论文"
    
    return title

def qwen_chat(messages):
    """使用Qwen LLM进行对话"""
    try:
        # 确保消息格式正确
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_messages.append(msg)
            else:
                print(f"跳过无效消息格式: {msg}")
                continue
        
        if not formatted_messages:
            return "错误：没有有效的消息格式"
        
        # 调用Qwen API
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=4000,
        )
        
        # 提取回复内容
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "错误：API返回空响应"
            
    except Exception as e:
        print(f"Qwen LLM调用失败：{e}")
        return f"LLM调用失败：{e}"

def web_scraper(state: PPTState):
    """爬取URL的文字内容和图片URL"""
    url = state["content_url"]
    
    try:
        # 发送HTTP请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 解析HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 提取文字内容（去除script和style标签）
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取所有文本内容
        text_content = soup.get_text()
        # 清理文本（去除多余空白）
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)
        
        # 提取图片URL - 改进版本
        image_urls = []
        
        # 处理arXiv网站的图片
        if 'arxiv.org' in url:
            # arXiv的图片通常在特定的位置
            # 查找所有可能的图片元素
            img_elements = soup.find_all('img')
            
            for img in img_elements:
                src = img.get('src')
                if src:
                    # 处理相对路径
                    if src.startswith('/'):
                        # 对于arXiv，图片通常在 /html/ 路径下
                        if '/html/' in src:
                            absolute_url = f"https://arxiv.org{src}"
                        else:
                            absolute_url = urljoin(url, src)
                    elif src.startswith('http'):
                        absolute_url = src
                    else:
                        absolute_url = urljoin(url, src)
                    
                    # 验证URL是否为有效的图片链接
                    if is_valid_image_url(absolute_url):
                        image_urls.append(absolute_url)
            
            # 如果没有找到图片，尝试从文本中提取图片引用
            if not image_urls:
                # 查找文本中的图片引用，如 "Figure 1", "Fig. 2" 等
                import re
                figure_patterns = [
                    r'Figure\s+\d+',
                    r'Fig\.\s*\d+',
                    r'图\s*\d+',
                    r'图表\s*\d+'
                ]
                
                figures_found = []
                for pattern in figure_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    figures_found.extend(matches)
                
                if figures_found:
                    print(f"发现图片引用：{figures_found[:5]}...")
        else:
            # 对于其他网站，使用通用方法
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    absolute_url = urljoin(url, src)
                    if is_valid_image_url(absolute_url):
                        image_urls.append(absolute_url)
        
        # 去重并限制数量
        image_urls = list(set(image_urls))[:100]  # 最多10张图片
        
        # 保存爬取结果到本地文件
        scraped_data = {
            "url": url,
            "text_content": text_content,
            "image_urls": image_urls,
            "timestamp": str(datetime.datetime.now())
        }
        
        with open("scraped_data.json", "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, ensure_ascii=False, indent=2)
        
        print(f"爬取完成！文字内容长度：{len(text_content)}字符，图片数量：{len(image_urls)}")
        if image_urls:
            print("图片链接示例：")
            for i, img_url in enumerate(image_urls[:3], 1):
                print(f"  {i}. {img_url}")
        print(f"数据已保存到：{os.path.abspath('scraped_data.json')}")
        
        return {
            "scraped_text": text_content,
            "image_urls": image_urls
        }
        
    except Exception as e:
        print(f"爬取失败：{e}")
        return {
            "scraped_text": f"爬取失败：{e}",
            "image_urls": []
        }

def arxiv_png_crawler(state: PPTState):
    """爬取URL的PNG图片"""
    url = state["content_url"]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml'
    }
    
    if not url.startswith(('http://', 'https://')):
        return {"png_images": [], "temp_filename": None}
    
    max_retries = 3
    timeout = 15
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tags = soup.find_all('img', {'src': lambda x: x and x.lower().endswith('.png')})
            
            base_url = url if url.endswith('/') else url + '/'
            png_urls = [urljoin(base_url, img['src']) for img in img_tags]
            
            # 去重
            png_urls = list(dict.fromkeys(png_urls))
            
            print(f"爬取到 {len(png_urls)} 张PNG图片")
            
            # 生成临时JSON文件存储图片信息
            temp_images_data = {
                "url": url,
                "png_images": png_urls,
                "image_count": len(png_urls),
                "crawl_time": str(datetime.datetime.now()),
                "image_details": []
            }
            
            # 为每张图片添加详细信息
            for i, img_url in enumerate(png_urls, 1):
                img_info = {
                    "index": i,
                    "url": img_url,
                    "filename": os.path.basename(img_url),
                    "status": "pending"  # 可以后续添加下载状态
                }
                temp_images_data["image_details"].append(img_info)
            
            # 保存临时JSON文件
            temp_filename = f"temp_images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump(temp_images_data, f, ensure_ascii=False, indent=2)
            
            print(f"图片信息已保存到临时文件：{temp_filename}")
            
            return {"png_images": png_urls, "temp_filename": temp_filename}
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"爬取PNG图片失败：{e}")
                return {"png_images": [], "temp_filename": None}
            time.sleep(retry_delay)
    
    return {"png_images": [], "temp_filename": None}

def is_valid_image_url(url: str) -> bool:
    """验证URL是否为有效的图片链接"""
    # 检查URL是否包含常见的图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
    url_lower = url.lower()
    
    # 检查是否包含图片扩展名
    has_image_ext = any(ext in url_lower for ext in image_extensions)
    
    # 检查是否包含图片相关的路径
    image_paths = ['/image', '/img', '/picture', '/photo', '/figure', '/fig']
    has_image_path = any(path in url_lower for path in image_paths)
    
    # 排除一些明显不是图片的URL
    exclude_patterns = ['logo', 'icon', 'avatar', 'button', 'banner']
    is_excluded = any(pattern in url_lower for pattern in exclude_patterns)
    
    return (has_image_ext or has_image_path) and not is_excluded

def text_summarizer(state: PPTState):
    """使用LLM总结文字内容"""
    text_content = state["scraped_text"]
    
    if not text_content or "爬取失败" in text_content:
        return {"text_summary": "无法总结：内容爬取失败"}
    
    # 如果内容太长，截取前5000字符
    if len(text_content) > 5000:
        text_content = text_content[:5000] + "...[内容已截取]"
    
    system_prompt = '''你是一个学术论文总结助手。请用中文总结论文内容，要求如下：
1. 包含基本信息：标题、作者、机构等
2. 按章节结构总结，每部分约200字
3. 保持客观性，使用学术语言
4. 避免数学公式和代码块
5. 标注重要数据和图表引用
6. 在最后注明原始来源

格式要求：
- 使用二级标题（##）划分章节
- 保持关键术语的英文原文，必要时加中文翻译
- 以要点形式呈现核心贡献
- 保持句子简洁，去除冗余论证
- 在最后添加"来源：论文标题"'''

    user_prompt = f"""请基于以下论文内容生成结构化的中文摘要：

{text_content}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    summary = qwen_chat(messages)
    
    return {"text_summary": summary}

def generate_paper_introduction(url: str) -> str:
    """主函数：输入论文链接，生成论文介绍"""
    print(f"开始处理论文链接：{url}")
    
    # 初始化状态
    state = PPTState(
        content_url=url,
        scraped_text="",
        image_urls=[],
        text_summary="",
        paper_title="未知论文", # 初始化论文标题
        png_images=[], # 初始化PNG图片列表
        temp_filename="" # 初始化临时文件名
    )
    
    # 第一步：爬取网页内容
    print("正在爬取网页内容...")
    scraped_result = web_scraper(state)
    state.update(scraped_result)
    
    # 第二步：爬取PNG图片
    print("正在爬取PNG图片...")
    png_result = arxiv_png_crawler(state)
    state.update(png_result)

    # 第三步：生成总结
    print("正在生成论文总结...")
    summary_result = text_summarizer(state)
    state.update(summary_result)
    
    # 提取论文标题
    state["paper_title"] = extract_paper_title(state["scraped_text"], url)
    print(f"提取到的论文标题：{state['paper_title']}")

    # 保存最终结果到JSON
    final_result = {
        "url": url,
        "summary": state["text_summary"],
        "image_count": len(state["image_urls"]),
        "png_image_count": len(state["png_images"]),
        "total_images": len(state["image_urls"]) + len(state["png_images"]),
        "image_urls": state["image_urls"],
        "png_images": state["png_images"],
        "timestamp": str(datetime.datetime.now())
    }
    
    # 创建paper_output文件夹
    output_dir = "paper_output"
    os.makedirs(output_dir, exist_ok=True)

    # 使用论文标题作为文件名
    output_filename = f"{state['paper_title']}.json"
    with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    # 生成HTML报告
    html_path = generate_html_report(url, state["text_summary"], state["image_urls"], state["scraped_text"], state["paper_title"], state["png_images"])
    
    # 删除临时文件
    if state.get("temp_filename"):
        try:
            os.remove(state["temp_filename"])
            print(f"临时文件 {state['temp_filename']} 已删除。")
        except OSError as e:
            print(f"删除临时文件失败：{e}")
    
    print(f"\n论文总结已保存到：{os.path.abspath(os.path.join(output_dir, output_filename))}")
    print(f"HTML报告已保存到：{os.path.abspath(html_path)}")
    print("\n" + "="*50)
    print("论文总结：")
    print("="*50)
    print(state["text_summary"])
    
    return state["text_summary"]

def generate_html_report(url: str, summary: str, image_urls: List[str], original_text: str, paper_title: str = "未知论文", png_images: List[str] = None):
    """生成包含图片的HTML报告"""
    
    if png_images is None:
        png_images = []
    
    # 生成图片HTML - 只使用PNG图片
    images_html = ""
    
    if png_images:
        images_html = '<div class="images-section">\n<h2>📷 论文PNG图片</h2>\n<div class="image-gallery">\n'
        
        # 显示PNG图片（限制20张）
        for i, img_url in enumerate(png_images[:20], 1):
            # 尝试从URL中提取图片名称
            img_name = f"PNG图片 {i}"
            
            # 从URL路径中提取文件名
            path_parts = img_url.split('/')
            if len(path_parts) > 1:
                filename = path_parts[-1]
                if '.' in filename:
                    img_name = filename.split('.')[0]
            
            # 尝试从文件名中提取图片编号
            import re
            fig_match = re.search(r'x(\d+)', img_name)
            if fig_match:
                img_name = f"Figure {fig_match.group(1)}"
            
            images_html += f'''
            <div class="image-item">
                <img src="{img_url}" alt="{img_name}" onerror="this.style.display='none'; this.nextElementSibling.innerHTML='图片加载失败'">
                <p class="image-caption">{img_name} (PNG)</p>
            </div>'''
        images_html += '</div>\n</div>\n'
    
    # 生成HTML内容
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>论文阅读报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }}
        .meta-info {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #007acc;
        }}
        .meta-info p {{
            margin: 5px 0;
        }}
        .summary-section {{
            margin-bottom: 30px;
        }}
        .summary-section h2 {{
            color: #007acc;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .summary-content {{
            background: #fafafa;
            padding: 20px;
            border-radius: 5px;
            white-space: pre-wrap;
            line-height: 1.8;
        }}
        .images-section {{
            margin-top: 30px;
        }}
        .images-section h2 {{
            color: #007acc;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-item {{
            text-align: center;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .image-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .image-caption {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        .original-text {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }}
        .original-text h3 {{
            color: #007acc;
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            .image-gallery {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 论文阅读报告</h1>
            <p>AI智能论文分析工具生成</p>
        </div>
        
        <div class="meta-info">
            <p><strong>📄 论文链接：</strong><a href="{url}" target="_blank">{url}</a></p>
            <p><strong>📅 生成时间：</strong>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-section">
            <h2>📝 论文摘要</h2>
            <div class="summary-content">{summary}</div>
        </div>
        
        {images_html}
        
        <div class="original-text">
            <h3>📄 原始文本（前1000字符）</h3>
            <p>{original_text[:1000]}{'...' if len(original_text) > 1000 else ''}</p>
        </div>
        
        <div class="footer">
            <p>由论文阅读工具生成 | 基于Qwen AI技术</p>
        </div>
    </div>
</body>
</html>'''
    
    # 创建paper_output文件夹
    output_dir = "paper_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用论文标题作为HTML文件名
    html_filename = f"{paper_title}.html"
    html_path = os.path.join(output_dir, html_filename)
    
    # 保存HTML文件
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path

if __name__ == "__main__":
    # 示例使用
    paper_url = input("请输入论文链接：")
    if paper_url.strip():
        generate_paper_introduction(paper_url)
    else:
        print("未输入有效链接")

