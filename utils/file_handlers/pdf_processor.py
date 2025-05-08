import os
import fitz  # PyMuPDF
from typing import List, Dict
import re

class PDFProcessor:
    def __init__(self, cache_dir: str, data_dir: str):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        
    def process_pdf(self, filename: str) -> Dict:
        """处理PDF文件并返回处理后的文本内容，将内容保存为TXT格式"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
            
        try:
            doc = fitz.open(file_path)
            text_content = []
            metadata = {
                "pages": len(doc),
                "title": doc.metadata.get("title", filename),
                "author": doc.metadata.get("author", "Unknown"),
                "original_format": "pdf"
            }
            
            for page in doc:
                text = page.get_text()
                # 清理文本
                text = self._clean_text(text)
                text_content.append(text)
                
            doc.close()
            
            # 生成处理后的文件名（转换为txt格式）
            base_name = os.path.splitext(filename)[0]
            txt_filename = f"{base_name}.txt"
            processed_path = os.path.join(self.data_dir, txt_filename)
            
            # 将处理后的内容保存为txt文件
            with open(processed_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(text_content))
            
            # 删除原始缓存文件
            os.remove(file_path)
            
            return {
                "content": "\n".join(text_content),
                "metadata": metadata,
                "file_path": processed_path
            }
            
        except Exception as e:
            raise Exception(f"处理PDF文件时出错: {str(e)}")
            
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text.strip()