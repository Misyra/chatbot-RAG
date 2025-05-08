import os
import json
from typing import Dict, Any
import re

class JSONProcessor:
    def __init__(self, cache_dir: str, data_dir: str):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        
    def process_json(self, filename: str) -> Dict:
        """处理JSON文件并返回处理后的内容，将内容保存为TXT格式"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON文件不存在: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            # 清理JSON内容
            cleaned_content = self._clean_json(content)
            
            # 将JSON对象转换为字符串表示
            if isinstance(cleaned_content, (dict, list)):
                text_content = json.dumps(cleaned_content, ensure_ascii=False, indent=2)
            else:
                text_content = str(cleaned_content)
            
            # 生成处理后的文件名（转换为txt格式）
            base_name = os.path.splitext(filename)[0]
            txt_filename = f"{base_name}.txt"
            processed_path = os.path.join(self.data_dir, txt_filename)
            
            # 将处理后的内容保存为txt文件
            with open(processed_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # 删除原始缓存文件
            os.remove(file_path)
            
            return {
                "content": text_content,
                "metadata": {
                    "title": base_name,
                    "file_size": os.path.getsize(processed_path),
                    "original_format": "json"
                },
                "file_path": processed_path
            }
            
        except Exception as e:
            raise Exception(f"处理JSON文件时出错: {str(e)}")
            
    def _clean_json(self, data: Any) -> Any:
        """清理JSON数据"""
        if isinstance(data, str):
            # 保留小数点等常用符号
            return re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()\[\]{}"\'-]', '', data).strip()
        elif isinstance(data, dict):
            # 递归清理字典
            return {k: self._clean_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            # 递归清理列表
            return [self._clean_json(item) for item in data]
        else:
            return data