import os
from typing import Dict
import re

class MDProcessor:
    def __init__(self, cache_dir: str, data_dir: str):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        
    def process_md(self, filename: str) -> Dict:
        """处理Markdown文件并返回处理后的文本内容"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown文件不存在: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 清理文本
            cleaned_content = self._clean_text(content)
            
            # 生成处理后的文件名（转换为txt格式）
            base_name = os.path.splitext(filename)[0]
            txt_filename = f"{base_name}.txt"
            processed_path = os.path.join(self.data_dir, txt_filename)
            
            # 将处理后的内容保存为txt文件
            with open(processed_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            # 删除原始缓存文件
            os.remove(file_path)
            
            return {
                "content": cleaned_content,
                "metadata": {
                    "title": base_name,
                    "file_size": os.path.getsize(processed_path),
                    "original_format": "markdown"
                },
                "file_path": processed_path
            }
            
        except Exception as e:
            raise Exception(f"处理Markdown文件时出错: {str(e)}")
            
    def _clean_text(self, text: str) -> str:
        """清理Markdown文本内容，移除Markdown标记"""
        # 移除标题标记 (#)
        text = re.sub(r'#+\s+', '', text)
        # 移除粗体和斜体标记 (* _ **)
        text = re.sub(r'[\*_]{1,2}([^\*_]+)[\*_]{1,2}', r'\1', text)
        # 移除链接 [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 移除图片 ![alt](url)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        # 移除代码块
        text = re.sub(r'```[\s\S]*?```', '', text)
        # 移除行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 移除水平线
        text = re.sub(r'---+', '', text)
        # 移除列表标记
        text = re.sub(r'^[\*\-+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # 移除引用标记
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 保留小数点等常用符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()\[\]{}"\'-]', '', text)
        return text.strip()