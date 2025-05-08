"""混合检索器模块，提供向量检索和BM25检索的混合功能"""

from typing import List, Dict, Optional, Union, Tuple
import os
import logging
import traceback
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import CrossEncoder
from .vector_store import VectorStore
from .bm25_retriever import EnhancedBM25Retriever
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HybridRetriever")

class HybridRetriever:
    """混合检索器，结合向量检索和BM25检索"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        use_reranker: bool = True,
        rerank_model: str = "BAAI/bge-reranker-base",
        debug_mode: bool = False,
        log_file: str = None
    ):
        """初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            use_reranker: 是否使用重排序
            rerank_model: 重排序模型名称
            debug_mode: 是否启用调试模式
            log_file: 日志文件路径
        """
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.rerank_model = rerank_model
        self.debug_mode = debug_mode
        self.reranker = None  # 改为延迟加载
        
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
        
        # 初始化BM25检索器
        self.bm25_retriever = EnhancedBM25Retriever(debug_mode=debug_mode, log_file=log_file)
        
        # 初始化文档
        self._initialize_documents()

    def _initialize_documents(self) -> None:
        """初始化文档到BM25检索器"""
        try:
            # 从向量存储获取所有文档
            documents = self.vector_store.get_documents()
            if not documents:
                return
                
            # 提取文本和元数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 添加到BM25检索器
            self.bm25_retriever.add_documents(texts, metadatas)
            logger.info(f"成功初始化 {len(texts)} 个文档到BM25检索器")
            
        except Exception as e:
            logger.error(f"初始化文档失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"初始化失败详情: {traceback.format_exc()}")

    def _load_reranker_if_needed(self):
        """按需加载重排序模型"""
        if self.use_reranker and self.reranker is None:
            try:
                # 使用 CrossEncoder 替代 SentenceTransformer
                logger.info(f"正在加载重排序模型: {self.rerank_model}")
                self.reranker = CrossEncoder(self.rerank_model)
                logger.info(f"成功加载重排序模型: {self.rerank_model}")
            except Exception as e:
                logger.error(f"加载重排序模型失败: {str(e)}")
                if self.debug_mode:
                    logger.debug(f"加载失败详情: {traceback.format_exc()}")
                self.use_reranker = False

    def _calculate_combined_score(self, vector_score: float, bm25_score: float) -> float:
        """计算综合得分
        
        Args:
            vector_score: 向量检索得分
            bm25_score: BM25检索得分
            
        Returns:
            综合得分
        """
        # 向量检索权重0.5，BM25权重0.5
        return 0.5 * vector_score + 0.5 * bm25_score

    def _rerank_documents(self, query: str, documents: List[Dict], k: int) -> List[Dict]:
        """使用CrossEncoder模型对文档进行重排序
        
        Args:
            query: 查询文本搜索完成搜索完成
            documents: 文档列表
            k: 返回结果数量
            
        Returns:
            重排序后的文档列表
        """
        if not self.use_reranker or not documents:
            logger.debug("跳过重排序：use_reranker=False 或 documents为空")
            return documents[:k]
            
        try:
            # 延迟加载重排序模型
            self._load_reranker_if_needed()
            
            # 如果加载失败则跳过重排序
            if not self.reranker:
                logger.warning("重排序模型未加载，跳过重排序")
                return documents[:k]
                
            # 记录重排序请求信息
            logger.debug(f"开始重排序:")
            logger.debug(f"- 重排序模型: {self.rerank_model}")
            logger.debug(f"- 查询文本: {query[:100]}...")
            logger.debug(f"- 文档数量: {len(documents)}")
            
            # 准备文档文本
            passages = [doc["content"] for doc in documents]
            
            # 准备查询-文档对
            pairs = [[query, passage] for passage in passages]
            
            # 计算相关性分数
            scores = self.reranker.predict(pairs)
            
            # 更新文档分数和排序
            for doc, score in zip(documents, scores):
                # 将重排序分数与原始分数结合
                original_score = doc.get("score", 0)
                new_score = 0.7 * float(score) + 0.3 * original_score
                doc["score"] = new_score
                logger.debug(f"文档分数更新: 原始={original_score:.4f}, 重排序={float(score):.4f}, 最终={new_score:.4f}")
            
            # 按分数排序
            reranked_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
            logger.info(f"重排序完成，返回前{k}个结果")
            
            return reranked_docs[:k]
                
        except Exception as e:
            logger.warning(f"重排序失败: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return documents[:k]

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """混合检索文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 扩大检索数量
            vector_k = 12  # 向量检索数量
            bm25_k = 12    # BM25检索数量
            
            # 向量检索
            vector_results = self.vector_store.similarity_search(query, k=vector_k)
            
            # BM25检索
            logger.info(f"执行BM25检索，查询: {query[:50]}...")
            bm25_results = self.bm25_retriever.search(query, k=bm25_k)
            logger.info(f"BM25检索完成，找到 {len(bm25_results)} 个结果")
            
            # 合并结果
            logger.info(f"开始合并向量检索和BM25检索结果")
            all_results = []
            seen_contents = set()
            
            # 添加向量检索结果
            for result in vector_results:
                content = result["content"]
                if content not in seen_contents:
                    seen_contents.add(content)
                    all_results.append({
                        "content": content,
                        "metadata": result["metadata"],
                        "vector_score": result["score"],
                        "bm25_score": 0.0,
                        "score": result["score"]  # 初始分数
                    })
            
            # 添加BM25检索结果并更新分数
            for result in bm25_results:
                content = result["content"]
                if content not in seen_contents:
                    seen_contents.add(content)
                    all_results.append({
                        "content": content,
                        "metadata": result["metadata"],
                        "vector_score": 0.0,
                        "bm25_score": result["score"],
                        "score": result["score"]  # 初始分数
                    })
                else:
                    # 更新已存在文档的BM25分数
                    for doc in all_results:
                        if doc["content"] == content:
                            doc["bm25_score"] = result["score"]
                            # 更新综合分数
                            doc["score"] = self._calculate_combined_score(
                                doc["vector_score"],
                                result["score"]
                            )
            
            # 按综合分数排序
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 重排序
            if self.use_reranker:
                all_results = self._rerank_documents(query, all_results, k)
            
            return all_results[:k]
            
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"检索失败详情: {traceback.format_exc()}")
            return []

    def get_stats(self) -> Dict:
        """获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        vector_stats = self.vector_store.get_stats()
        bm25_stats = self.bm25_retriever.get_stats()
        
        return {
            "vector_store": vector_stats,
            "bm25_retriever": bm25_stats,
            "use_reranker": self.use_reranker,
            "rerank_model": self.rerank_model,
            "reranker_loaded": self.reranker is not None,
            "status": "就绪"
        }

    def update_documents(self) -> None:
        """更新文档"""
        try:
            # 清空BM25检索器
            self.bm25_retriever.clear()
            
            # 重新初始化文档
            self._initialize_documents()
            
            logger.info("文档更新完成")
            
        except Exception as e:
            logger.error(f"更新文档失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"更新失败详情: {traceback.format_exc()}")