"""BM25检索器模块，提供基于BM25算法的文本检索功能"""

from typing import List, Dict, Optional, Union, Tuple
import os
import logging
import traceback
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BM25Retriever")

class EnhancedBM25Retriever:
    """增强的BM25检索器，支持中文分词和文档管理"""
    
    def __init__(self, debug_mode: bool = False, log_file: str = None):
        """初始化BM25检索器
        
        Args:
            debug_mode: 是否启用调试模式
            log_file: 日志文件路径
        """
        self.debug_mode = debug_mode
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        
        # 配置日志
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
        
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        # 初始化jieba分词
        jieba.initialize()

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        try:
            return list(jieba.cut(text))
        except Exception as e:
            logger.error(f"分词失败: {str(e)}")
            return list(text)

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """添加文档到检索器
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        if not texts:
            return
            
        try:
            # 准备文档
            tokenized_texts = [self._tokenize(text) for text in texts]
            
            # 更新文档列表
            self.documents.extend(texts)
            
            # 更新元数据
            if metadatas:
                self.metadatas.extend(metadatas)
            else:
                self.metadatas.extend([{
                    "source": f"text_{i}",
                    "timestamp": datetime.now().isoformat()
                } for i in range(len(texts))])
            
            # 创建或更新BM25模型
            if self.bm25 is None:
                self.bm25 = BM25Okapi(tokenized_texts)
            else:
                self.bm25 = BM25Okapi(tokenized_texts + [self._tokenize(doc) for doc in self.documents])
            
            logger.info(f"成功添加 {len(texts)} 个文档")
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"添加失败详情: {traceback.format_exc()}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.bm25 or not self.documents:
            return []
            
        try:
            # 对查询进行分词
            tokenized_query = self._tokenize(query)
            
            # 计算文档分数
            scores = self.bm25.get_scores(tokenized_query)
            
            # 获取前k个结果
            top_indices = np.argsort(scores)[-k:][::-1]
            
            # 构建结果
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只返回有相关性的结果
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadatas[idx],
                        "score": float(scores[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"搜索失败详情: {traceback.format_exc()}")
            return []

    def get_stats(self) -> Dict:
        """获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "doc_count": len(self.documents),
            "status": "就绪" if self.bm25 is not None else "未初始化"
        }

    def get_documents(self) -> List[Dict]:
        """获取所有文档
        
        Returns:
            文档列表
        """
        return [
            {
                "content": doc,
                "metadata": meta
            }
            for doc, meta in zip(self.documents, self.metadatas)
        ]

    def clear(self) -> None:
        """清空检索器"""
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        logger.info("检索器已清空") 