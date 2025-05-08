import os
import mimetypes
import logging
from typing import Dict, List, Optional
from .pdf_processor import PDFProcessor
from .txt_processor import TXTProcessor
from .json_processor import JSONProcessor
from .md_processor import MDProcessor
from .general_processor import GeneralProcessor

# 获取logger
logger = logging.getLogger("陶瓷问答系统.FileManager")

class FileManager:
    def __init__(self, cache_dir: str, data_dir: str):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.custom_files_dir = os.path.join(data_dir, "custom_files")
        # 确保目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.custom_files_dir, exist_ok=True)
        
        # 初始化各种文件处理器
        self.pdf_processor = PDFProcessor(cache_dir, self.custom_files_dir)
        self.txt_processor = TXTProcessor(cache_dir, self.custom_files_dir)
        self.json_processor = JSONProcessor(cache_dir, self.custom_files_dir)
        self.md_processor = MDProcessor(cache_dir, self.custom_files_dir)
        self.general_processor = GeneralProcessor(cache_dir, self.custom_files_dir)
        
        # 确保目录存在
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
    def process_file(self, filename: str) -> Dict:
        """根据文件类型处理文件，将所有文件处理为txt格式保存到custom_files目录，处理完成后删除缓存文件"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 确保custom_files目录存在
        os.makedirs(self.custom_files_dir, exist_ok=True)
        
        # 根据文件扩展名处理不同类型的文件
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            result = None
            if file_ext == '.pdf':
                result = self.pdf_processor.process_pdf(filename)
            elif file_ext == '.txt':
                result = self.txt_processor.process_txt(filename)
            elif file_ext == '.json':
                result = self.json_processor.process_json(filename)
            elif file_ext == '.md':
                result = self.md_processor.process_md(filename)
            elif file_ext in ['.csv', '.docx']:
                # 使用通用处理器处理CSV和DOCX文件
                result = self.general_processor.process_file(filename)
            else:
                # 对于其他不支持的文件类型，使用通用处理器
                result = self.general_processor.process_file(filename)
                
            # 处理成功后，删除缓存文件
            if result and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"已从缓存中删除文件: {filename}")
                except Exception as del_e:
                    logger.error(f"删除缓存文件时出错: {str(del_e)}")
                    
            return result
        except Exception as e:
            # 如果处理失败，尝试使用通用处理器
            try:
                result = self.general_processor.process_file(filename)
                
                # 处理成功后，删除缓存文件
                if result and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"已从缓存中删除文件: {filename}")
                    except Exception as del_e:
                        logger.error(f"删除缓存文件时出错: {str(del_e)}")
                        
                return result
            except Exception as inner_e:
                raise Exception(f"处理文件失败: {str(e)}，尝试通用处理器也失败: {str(inner_e)}")
            
    def list_files(self, directory: str) -> List[Dict]:
        """列出指定目录中的文件信息"""
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                files.append({
                    "name": filename,
                    "size": os.path.getsize(file_path),
                    "modified_time": os.path.getmtime(file_path)
                })
        return files
        
    def delete_file(self, filename: str, directory: str) -> bool:
        """删除指定目录中的文件"""
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
        
    def rename_file(self, old_name: str, new_name: str, directory: str) -> bool:
        """重命名指定目录中的文件"""
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            return True
        return False
        
    def get_file_info(self, filename: str, directory: str) -> Optional[Dict]:
        """获取文件详细信息"""
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return {
                "name": filename,
                "size": os.path.getsize(file_path),
                "modified_time": os.path.getmtime(file_path),
                "created_time": os.path.getctime(file_path)
            }
        return None