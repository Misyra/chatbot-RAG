import os
import json
from typing import Dict, Optional
import hashlib
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, cache_dir: str, max_size: int = 1000, ttl: int = 3600):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = ttl  # 缓存过期时间（秒）
        self.cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()
        
    def _load_cache(self) -> None:
        """加载缓存文件"""
        cache_file = os.path.join(self.cache_dir, "cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
                
    def _save_cache(self) -> None:
        """保存缓存到文件"""
        cache_file = os.path.join(self.cache_dir, "cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
            
    def _get_cache_key(self, query: str) -> str:
        """生成缓存键"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
        
    def get(self, query: str) -> Optional[Dict]:
        """获取缓存结果"""
        key = self._get_cache_key(query)
        if key in self.cache:
            cache_item = self.cache[key]
            # 检查是否过期
            if datetime.fromisoformat(cache_item['timestamp']) + timedelta(seconds=self.ttl) > datetime.now():
                return cache_item['result']
            else:
                del self.cache[key]
        return None
        
    def set(self, query: str, result: Dict) -> None:
        """设置缓存结果"""
        key = self._get_cache_key(query)
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        # 如果超过最大大小，删除最旧的缓存
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.cache[oldest_key]
            
        self._save_cache()
        
    def clear(self) -> None:
        """清空缓存"""
        self.cache = {}
        self._save_cache()
        
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        } 