"""向量存储模块，提供文档的向量化和检索功能"""

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
import numpy as np
from tqdm import tqdm
import logging
import sys
import traceback
from datetime import datetime
import faiss
import pickle
import requests
import hashlib
import functools
import json
import dataclasses
from pathlib import Path

torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)
]  # Fix for torch classes not found error

# 设置OpenMP环境变量以解决冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VectorStore")

class PerformanceMonitor:
    """性能监控类，用于记录和统计操作耗时"""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
        
    def record_operation(self, operation: str, duration: float):
        """记录操作耗时
        
        Args:
            operation: 操作名称
            duration: 耗时（秒）
        """
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
        
    def get_stats(self) -> Dict:
        """获取性能统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        for operation in self.operation_times:
            times = self.operation_times[operation]
            stats[operation] = {
                "count": self.operation_counts[operation],
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "total_time": np.sum(times)
            }
        return stats

# 创建一个嵌入缓存装饰器
def cache_embeddings(func):
    """缓存嵌入向量，避免重复生成"""
    embeddings_cache = {}
    cache_stats = {"hits": 0, "misses": 0}

    @functools.wraps(func)
    def wrapper(self, texts, **kwargs):
        # 对于单个文本的特殊处理
        if isinstance(texts, str):
            cache_key = hashlib.md5(texts.encode('utf-8')).hexdigest()
            if cache_key in embeddings_cache:
                cache_stats["hits"] += 1
                logger.debug(f"嵌入缓存命中: {cache_key[:8]}...")
                return embeddings_cache[cache_key]
            cache_stats["misses"] += 1
            result = func(self, texts, **kwargs)
            embeddings_cache[cache_key] = result
            return result

        # 对于文本列表的处理
        results = []
        new_texts = []
        indices = []

        # 检查哪些文本需要重新嵌入
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
            if cache_key in embeddings_cache:
                cache_stats["hits"] += 1
                results.append((i, embeddings_cache[cache_key]))
            else:
                cache_stats["misses"] += 1
                new_texts.append(text)
                indices.append(i)

        # 如果有需要重新嵌入的文本
        if new_texts:
            new_embeddings = func(self, new_texts, **kwargs)
            for idx, embedding in zip(indices, new_embeddings):
                cache_key = hashlib.md5(texts[idx].encode('utf-8')).hexdigest()
                embeddings_cache[cache_key] = embedding
                results.append((idx, embedding))

        # 记录缓存统计信息
        hit_rate = cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
        logger.debug(f"嵌入缓存统计 - 命中率: {hit_rate:.2%}, 命中: {cache_stats['hits']}, 未命中: {cache_stats['misses']}")

        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        return [embedding for _, embedding in results]

    return wrapper

class FallbackEmbeddings:
    """备用嵌入模型，在Ollama API无法使用时生成一致性的伪嵌入向量"""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        logger.warning(f"初始化FallbackEmbeddings，维度: {dimension}")

    def _get_text_hash(self, text: str) -> int:
        """获取文本的哈希值，用于生成伪随机但稳定的嵌入向量"""
        return int(hash(text.encode("utf-8")).hexdigest(), 16)

    def _generate_fake_embedding(self, text: str) -> List[float]:
        """根据文本哈希生成假嵌入向量"""
        text_hash = self._get_text_hash(text)
        valid_seed = text_hash % (2**32)
        np.random.seed(valid_seed)
        vector = np.random.normal(0, 1, self.dimension)
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        return [self._generate_fake_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入向量"""
        return self._generate_fake_embedding(text)

# 扩展OllamaEmbeddings添加缓存
class CachedOllamaEmbeddings(OllamaEmbeddings):
    """添加嵌入缓存的OllamaEmbeddings"""

    @cache_embeddings
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts)

    @cache_embeddings
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text)

# 添加支持分数属性的文档类
@dataclasses.dataclass
class DocumentWithScore(Document):
    """支持添加分数属性的Document类"""
    score: float = 0.0
    
    def __init__(self, page_content: str, metadata: dict = None, score: float = 0.0):
        """初始化带分数的文档对象
        
        Args:
            page_content: 文档内容
            metadata: 文档元数据
            score: 相关度分数
        """
        super().__init__(page_content=page_content, metadata=metadata or {})
        self.score = score

class VectorStore:
    def __init__(
        self,
        storage_dir: str = "./storage/vector_store",
        embedding_model: str = None,
        base_url: str = "http://localhost:11434",
        debug_mode: bool = False,
        log_file: str = None,
        distance_strategy: str = "cosine"
    ):
        """初始化向量存储
        
        Args:
            storage_dir: 存储目录
            embedding_model: 嵌入模型名称
            base_url: Ollama服务地址
            debug_mode: 是否启用调试模式
            log_file: 日志文件路径
            distance_strategy: 距离计算策略
        """
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model or os.getenv("EMBEDDINGS_MODEL", "BGE-M3:latest")
        self.base_url = base_url
        self.distance_strategy = distance_strategy
        self.debug_mode = debug_mode
        self.vector_store = None
        self.performance_monitor = PerformanceMonitor()
        
        # 日志配置
        if log_file:
            log_dir = Path(log_file).parent
            if log_dir and not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
            
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            
        # 确保存储目录存在
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化嵌入模型
        try:
            self.ollama_available = self._test_ollama_connection()
            if self.ollama_available:
                logger.info(f"使用Ollama嵌入模型: {self.embedding_model}")
                self.embeddings = CachedOllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=base_url,
                )
            else:
                logger.warning("Ollama服务不可用，使用备用嵌入模型")
                self.embeddings = FallbackEmbeddings(dimension=1536)
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"初始化失败详情: {traceback.format_exc()}")
            self.embeddings = FallbackEmbeddings(dimension=1536)
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        )
        
        # 尝试加载已有的向量存储
        try:
            self.load()
        except Exception as e:
            logger.info("无法加载现有向量存储，系统将在首次使用时创建新的向量存储")
            if self.debug_mode:
                logger.debug(f"加载失败详情: {str(e)}")

    def _test_ollama_connection(self) -> bool:
        """测试Ollama API是否可用"""
        try:
            logger.info(f"测试Ollama API连接: {self.base_url}")
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if self.embedding_model in model_names:
                    logger.info(f"找到指定的嵌入模型: {self.embedding_model}")
                    return True
                # 如果没有找到指定的模型，尝试使用其他可用的嵌入模型
                available_models = [name for name in model_names if "bge" in name.lower() or "llama" in name.lower()]
                if available_models:
                    self.embedding_model = available_models[0]
                    logger.info(f"使用替代嵌入模型: {self.embedding_model}")
                    return True
                logger.warning("未找到合适的嵌入模型")
                return False
        except Exception as e:
            logger.error(f"测试Ollama API连接失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"连接失败详情: {traceback.format_exc()}")
            return False

    def create_from_texts(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """从文本创建向量存储"""
        if not texts:
            logger.warning("没有提供文本数据")
            return

        logger.info(f"开始从 {len(texts)} 个文本创建向量存储")
        start_time = time.time()

        # 准备文档
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"text_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))

        # 分割文档
        try:
            logger.info("开始分割文档...")
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"文档已分割为 {len(splits)} 个片段")
        except Exception as e:
            logger.error(f"分割文档时出错: {str(e)}")
            if self.debug_mode:
                logger.debug(f"分割失败详情: {traceback.format_exc()}")
            splits = documents
            
        # 创建向量存储
        try:
            logger.info("开始创建向量存储...")
            self.vector_store = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings,
                distance_strategy=self.distance_strategy
            )
            logger.info(f"向量存储创建完成，包含 {len(splits)} 个文档")
            self.save()
        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"创建失败详情: {traceback.format_exc()}")
            raise
    
        end_time = time.time()
        duration = end_time - start_time
        self.performance_monitor.record_operation("create_from_texts", duration)
        logger.info(f"向量存储创建完成，总耗时: {duration:.2f}秒")

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """添加文本到向量存储"""
        if not texts:
            return
            
        if self.vector_store is None:
            return self.create_from_texts(texts, metadatas)
            
        logger.info(f"开始添加 {len(texts)} 个文本到向量存储")
        start_time = time.time()
        
        # 准备文档
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"text_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
            
        # 分割文档
        try:
            logger.info("开始分割文档...")
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"文档已分割为 {len(splits)} 个片段")
        except Exception as e:
            logger.error(f"分割文档时出错: {str(e)}")
            if self.debug_mode:
                logger.debug(f"分割失败详情: {traceback.format_exc()}")
            splits = documents
            
        # 添加文档
        try:
            logger.info("开始添加文档到向量存储...")
            self.vector_store.add_documents(splits)
            logger.info(f"成功添加 {len(splits)} 个文档")
            self.save()
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"添加失败详情: {traceback.format_exc()}")
            raise
            
        end_time = time.time()
        duration = end_time - start_time
        self.performance_monitor.record_operation("add_texts", duration)
        logger.info(f"添加完成，总耗时: {duration:.2f}秒")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """相似度搜索"""
        if self.vector_store is None:
            return []
            
        try:
            logger.info(f"执行相似度搜索，查询: {query[:50]}...")
            start_time = time.time()
            
            docs = self.vector_store.similarity_search(query, k=k)
            results = []
            for i, doc in enumerate(docs):
                score = 1.0 - (i / len(docs)) if docs else 1.0
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
                
            end_time = time.time()
            duration = end_time - start_time
            self.performance_monitor.record_operation("similarity_search", duration)
            logger.info(f"搜索完成，找到 {len(results)} 个结果，耗时: {duration:.2f}秒")
            
            return results
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"搜索失败详情: {traceback.format_exc()}")
            return []

    def save(self, filename: str = None) -> None:
        """保存向量存储
        
        Args:
            filename: 已废弃参数，为保持兼容性保留，但不再使用
        """
        if self.vector_store is None:
            return
            
        try:
            logger.info("开始保存向量存储...")
            start_time = time.time()
            
            # 确保存储目录存在
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # 直接保存到storage目录
            index_path = self.storage_dir / "index.faiss"
            docstore_path = self.storage_dir / "docstore.pkl"
            
            # 保存索引
            if hasattr(self.vector_store, "index"):
                logger.debug("保存FAISS索引...")
                faiss.write_index(self.vector_store.index, str(index_path))
                
            # 保存文档存储
            if hasattr(self.vector_store, "docstore"):
                logger.debug("保存文档存储...")
                with open(docstore_path, "wb") as f:
                    pickle.dump(self.vector_store.docstore, f)
                    
            end_time = time.time()
            duration = end_time - start_time
            self.performance_monitor.record_operation("save", duration)
            logger.info(f"向量存储已保存到: {self.storage_dir}，耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"保存失败详情: {traceback.format_exc()}")
            raise

    def load(self, filename: str = None) -> None:
        """加载向量存储
        
        Args:
            filename: 已废弃参数，为保持兼容性保留，但不再使用
        """
        try:
            logger.info("开始加载向量存储...")
            start_time = time.time()
            
            # 直接从storage目录加载
            index_path = self.storage_dir / "index.faiss"
            docstore_path = self.storage_dir / "docstore.pkl"
            
            if not index_path.exists() or not docstore_path.exists():
                logger.info("未找到向量存储文件，将在首次使用时创建新的向量存储")
                return
                
            # 加载FAISS索引
            logger.debug("加载FAISS索引...")
            index = faiss.read_index(str(index_path))
            
            # 加载文档存储
            logger.debug("加载文档存储...")
            with open(docstore_path, "rb") as f:
                docstore = pickle.load(f)
                
            # 创建FAISS实例
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=list(docstore._dict.keys()) if hasattr(docstore, "_dict") else list(docstore.docs.keys()),
                normalize_L2=True,
                distance_strategy=self.distance_strategy
            )
            
            end_time = time.time()
            duration = end_time - start_time
            self.performance_monitor.record_operation("load", duration)
            logger.info(f"向量存储已加载，包含 {len(self.vector_store.index_to_docstore_id)} 个文档，耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"加载失败详情: {traceback.format_exc()}")
            raise

    def get_stats(self) -> Dict:
        """获取向量存储统计信息"""
        if self.vector_store is None:
            return {"doc_count": 0, "status": "未初始化"}
            
        try:
            doc_count = len(self.vector_store.index_to_docstore_id) if hasattr(self.vector_store, "index_to_docstore_id") else 0
            stats = {
                "doc_count": doc_count,
                "embedding_model": self.embedding_model if self.ollama_available else "备用嵌入",
                "status": "就绪",
                "distance_strategy": self.distance_strategy,
                "performance": self.performance_monitor.get_stats()
            }
            logger.debug(f"获取统计信息: {json.dumps(stats, indent=2)}")
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"获取统计信息失败详情: {traceback.format_exc()}")
            return {"doc_count": 0, "error": str(e), "status": "错误"}
            
    def get_texts(self) -> List[str]:
        """获取所有文本内容"""
        if self.vector_store is None:
            return []
            
        try:
            logger.debug("获取所有文本内容...")
            texts = []
            if hasattr(self.vector_store.docstore, "_dict"):
                for doc in self.vector_store.docstore._dict.values():
                    texts.append(doc.page_content)
            elif hasattr(self.vector_store.docstore, "docs"):
                for doc in self.vector_store.docstore.docs.values():
                    texts.append(doc.page_content)
            logger.debug(f"获取到 {len(texts)} 个文本")
            return texts
        except Exception as e:
            logger.error(f"获取文本内容失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"获取文本内容失败详情: {traceback.format_exc()}")
            return []
            
    def get_documents(self) -> List[Document]:
        """获取所有文档对象"""
        if self.vector_store is None:
            return []
            
        try:
            logger.debug("获取所有文档对象...")
            if hasattr(self.vector_store.docstore, "_dict"):
                docs = list(self.vector_store.docstore._dict.values())
            elif hasattr(self.vector_store.docstore, "docs"):
                docs = list(self.vector_store.docstore.docs.values())
            else:
                docs = []
            logger.debug(f"获取到 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.error(f"获取文档对象失败: {str(e)}")
            if self.debug_mode:
                logger.debug(f"获取文档对象失败详情: {traceback.format_exc()}")
            return []
