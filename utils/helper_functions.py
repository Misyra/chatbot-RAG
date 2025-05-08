import os
import fitz  # PyMuPDF
import json
from typing import List, Tuple, Optional
import re

def read_data_directory(data_dir: str) -> List[Tuple[str, Optional[dict]]]:
    """读取data目录下所有支持的文件内容和元数据（文件名）"""
    all_texts_metadata = []
    if not os.path.isdir(data_dir):
        print(f"警告：Data 目录不存在: {data_dir}")
        return []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(file_path):
            continue

        content = None
        metadata = {"source": filename} # 基础元数据
        try:
            if filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif filename.lower().endswith('.pdf'):
                doc = fitz.open(file_path)
                content = "\n".join([page.get_text() for page in doc])
                # 添加 PDF 元数据
                metadata["pages"] = len(doc)
                if doc.metadata:
                    metadata["title"] = doc.metadata.get("title", filename)
                    metadata["author"] = doc.metadata.get("author", "Unknown")
                doc.close()
            elif filename.lower().endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # 尝试提取所有字符串值作为内容
                    strings = []
                    def extract_strings(data):
                        if isinstance(data, str):
                            strings.append(data)
                        elif isinstance(data, dict):
                            for v in data.values():
                                extract_strings(v)
                        elif isinstance(data, list):
                            for item in data:
                                extract_strings(item)
                    extract_strings(json_data)
                    content = "\n".join(strings)
                    # metadata['json_structure'] = json.dumps(json_data, ensure_ascii=False, indent=2)[:500] # 可选：部分结构

            if content:
                # 清理文本
                content = re.sub(r'\s+', ' ', content).strip()

                content = re.sub(r'[^\w\s\u4e00-\u9fff,.!?;:"\'()\[\]{}<>《》【】（）“”‘’：；，。？！、-]', '', content)
                all_texts_metadata.append((content, metadata))

        except Exception as e:
            print(f"读取或处理文件 {filename} 时出错: {e}") # 打印错误而不是停止

    return all_texts_metadata