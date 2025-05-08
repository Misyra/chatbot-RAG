import os
from typing import Dict
import re

class TXTProcessor:
    def __init__(self, cache_dir: str, data_dir: str):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        
    def process_txt(self, filename: str) -> Dict:
        """处理TXT文件并返回处理后的文本内容"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TXT文件不存在: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 清理文本
            cleaned_content = self._clean_text(content)
            
            # 确保文件名是.txt格式
            if not filename.lower().endswith('.txt'):
                base_name = os.path.splitext(filename)[0]
                txt_filename = f"{base_name}.txt"
            else:
                txt_filename = filename
                
            # 将处理后的内容保存到data目录
            processed_path = os.path.join(self.data_dir, txt_filename)
            
            # 将处理后的内容写入新文件
            with open(processed_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
                
            # 删除原始缓存文件
            os.remove(file_path)
            
            return {
                "content": cleaned_content,
                "metadata": {
                    "title": os.path.splitext(txt_filename)[0],
                    "file_size": os.path.getsize(processed_path),
                    "original_format": "txt"
                },
                "file_path": processed_path
            }
            
        except Exception as e:
            raise Exception(f"处理TXT文件时出错: {str(e)}")
            
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 保留小数点等常用符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()\[\]{}"\'-]', '', text)
        return text.strip()