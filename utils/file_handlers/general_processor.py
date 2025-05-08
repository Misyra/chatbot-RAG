import os
from typing import Dict
import re
import csv
import io

class GeneralProcessor:
    def __init__(self, cache_dir: str, data_dir: str):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        
    def process_file(self, filename: str) -> Dict:
        """处理通用文件并返回处理后的文本内容"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        try:
            # 获取文件扩展名
            file_ext = os.path.splitext(filename)[1].lower()
            
            # 根据文件类型选择不同的处理方法
            if file_ext == '.csv':
                content = self._process_csv(file_path)
            elif file_ext == '.docx':
                content = self._process_docx(file_path)
            else:
                # 对于其他未知类型的文件，尝试以文本方式打开
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # 如果UTF-8解码失败，尝试其他编码
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            content = f.read()
                    except:
                        # 如果仍然失败，将文件视为二进制文件
                        content = f"[无法解析的二进制文件: {filename}]"
            
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
                    "original_format": file_ext[1:] if file_ext else "unknown"
                },
                "file_path": processed_path
            }
            
        except Exception as e:
            raise Exception(f"处理文件时出错: {str(e)}")
    
    def _process_csv(self, file_path: str) -> str:
        """处理CSV文件并返回文本内容"""
        try:
            text_content = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    text_content.append(', '.join(row))
            return '\n'.join(text_content)
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试GBK编码
            with open(file_path, 'r', encoding='gbk', newline='') as f:
                reader = csv.reader(f)
                text_content = []
                for row in reader:
                    text_content.append(', '.join(row))
            return '\n'.join(text_content)
    
    def _process_docx(self, file_path: str) -> str:
        """处理DOCX文件并返回文本内容"""
        try:
            # 尝试导入docx模块
            import docx
            doc = docx.Document(file_path)
            text_content = []
            for para in doc.paragraphs:
                text_content.append(para.text)
            return '\n'.join(text_content)
        except ImportError:
            return f"[需要安装python-docx库来处理DOCX文件: {os.path.basename(file_path)}]"
        except Exception as e:
            return f"[处理DOCX文件时出错: {str(e)}]"
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 保留中文、英文、数字和基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()\[\]{}"\'-]', '', text)
        return text.strip()