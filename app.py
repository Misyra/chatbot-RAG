"""
智能陶瓷问答系统 - 基于RAG的Streamlit Web应用
"""

import os
import sys
import streamlit as st
import traceback
from dotenv import load_dotenv
import logging
import threading
import time
from utils.file_handlers.file_manager import FileManager
from utils.graph.knowledge_graph import KnowledgeGraph
from utils.retrieval.vector_store import VectorStore
from utils.retrieval.retriever import HybridRetriever
from utils.cache.cache_manager import CacheManager
import requests
import json
from typing import Dict, List, Optional
import bcrypt
from datetime import datetime
from utils.helper_functions import read_data_directory  # 导入辅助函数
import torch
from openai import OpenAI


# 修复torch类找不到的错误
torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)
]  


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("陶瓷问答系统")

# 加载环境变量
load_dotenv()

# 从环境变量获取模型配置
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BGE-M3:latest")
DEBUG_MODE = os.getenv("DEBUG_MODE", "").lower() in ["true", "1", "yes", "y"]
LOG_FILE = os.getenv("LOG_FILE", "")

# 配置日志文件
if LOG_FILE:
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"已创建日志目录: {log_dir}")
        
        test_file = os.path.join(log_dir, "permission_test.log")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("权限测试\n")
        os.remove(test_file)

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        os.environ["VECTOR_STORE_LOG_FILE"] = LOG_FILE
    except Exception as e:
        logger.error(f"日志文件配置失败: {str(e)}")
        logger.error(f"请检查日志目录 {log_dir} 的写入权限")

# 设置调试模式
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    os.environ["VECTOR_STORE_DEBUG"] = "true"
    logger.debug("调试模式已启用")

# 打印关键配置信息
logger.info("系统配置: OLLAMA_API_URL=%s, OLLAMA_MODEL=%s, EMBEDDINGS_MODEL=%s, DEBUG_MODE=%s, LOG_FILE=%s",
            OLLAMA_API_URL, OLLAMA_MODEL, EMBEDDINGS_MODEL, DEBUG_MODE, LOG_FILE)

# 创建专用的prompt日志记录器
prompt_logger = logging.getLogger("prompt_logger")
prompt_logger.setLevel(logging.INFO)
if LOG_FILE:
    prompt_handler = logging.FileHandler(f"{LOG_FILE}.prompt", encoding="utf-8")
    prompt_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    prompt_logger.addHandler(prompt_handler)


# 配置页面
st.set_page_config(page_title="智能陶瓷问答系统", page_icon="🏺", layout="wide")

# 初始化目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DATA_DIR = os.path.join(BASE_DIR, "data")
CUSTOM_FILES_DIR = os.path.join(DATA_DIR, "custom_files")  # 用户上传的文件目录
DEFAULT_DATA_DIR = os.path.join(DATA_DIR, "default_data")  # 系统默认数据目录
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
QA_CACHE_DIR = os.path.join(CACHE_DIR, "qa_cache")

# 定义文件路径
GRAPH_PATH = os.path.join(STORAGE_DIR, "knowledge_graph.gpickle")
VECTOR_STORE_FILENAME = "vector_store.faiss"  # FAISS会自动处理目录

# 确保所有必需的目录和__init__.py文件存在
def ensure_directories_and_init_files():
    """确保所有必需的目录和__init__.py文件存在"""
    # 创建必需的目录
    for directory in [
        CACHE_DIR,
        DATA_DIR,
        CUSTOM_FILES_DIR,
        DEFAULT_DATA_DIR,
        STORAGE_DIR,
        QA_CACHE_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    # 确保utils目录结构完整
    utils_dirs = [
        os.path.join(BASE_DIR, "utils"),
        os.path.join(BASE_DIR, "utils", "file_handlers"),
        os.path.join(BASE_DIR, "utils", "graph"),
        os.path.join(BASE_DIR, "utils", "retrieval"),
        os.path.join(BASE_DIR, "utils", "cache"),
    ]

    # 创建目录和初始化文件
    for directory in utils_dirs:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# 自动生成的初始化文件\n")


# 执行目录和文件检查
ensure_directories_and_init_files()

# --- 组件初始化函数 ---
def init_components():
    """延迟初始化所有组件"""
    try:
        from utils.file_handlers.file_manager import FileManager
        from utils.retrieval.vector_store import VectorStore
        from utils.retrieval.retriever import HybridRetriever
        from utils.graph.knowledge_graph import KnowledgeGraph
        from utils.cache.cache_manager import CacheManager

        # 初始化组件
        logger.info("初始化系统组件...")

        # 初始化文件管理器
        file_manager = FileManager(CACHE_DIR, DATA_DIR)

        # 初始化缓存管理器
        cache_manager = CacheManager(QA_CACHE_DIR)

        # 初始化图谱管理器
        knowledge_graph = KnowledgeGraph()

        # 初始化向量存储
        vector_store = VectorStore(
            STORAGE_DIR,
            embedding_model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_API_URL,
            debug_mode=DEBUG_MODE,
            log_file=LOG_FILE,
        )

        # 尝试加载现有索引
        try:
            logger.info(f"尝试加载向量索引: {VECTOR_STORE_FILENAME}")
            vector_store.load(VECTOR_STORE_FILENAME)
            # 不要在后台线程中调用st.sidebar等UI组件
            logger.info("向量索引加载成功")
        except Exception as e:
            error_msg = f"未找到或加载向量索引失败: {e}"
            logger.warning(error_msg)
            if DEBUG_MODE:
                logger.debug(f"加载向量索引详细错误信息: {str(e)}")

        # 初始化混合检索器
        hybrid_retriever = HybridRetriever(vector_store)

        # 尝试加载知识图谱
        try:
            if os.path.exists(GRAPH_PATH):
                logger.info(f"尝试加载知识图谱: {GRAPH_PATH}")
                knowledge_graph.load_graph(GRAPH_PATH)
                logger.info("知识图谱加载成功")
            else:
                logger.info("未找到知识图谱文件")
        except Exception as e:
            error_msg = f"加载知识图谱时出错: {e}"
            logger.error(error_msg)

        # 更新会话状态
        # 不要直接在线程中操作st.session_state，改用返回值模式
        return {
            "file_manager": file_manager,
            "cache_manager": cache_manager,
            "knowledge_graph_instance": knowledge_graph,
            "vector_store_instance": vector_store,
            "hybrid_retriever_instance": hybrid_retriever,
            "components_loaded": True
        }

    except Exception as e:
        logger.error(f"初始化组件时出错: {e}")
        logger.debug(traceback.format_exc())
        return {"components_loaded": False, "error": str(e)}

# 初始全局组件
global_components = None

# --- 会话状态初始化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "max_contexts" not in st.session_state:
    st.session_state.max_contexts = 3  # 设置默认值为3
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7  # 设置默认值
if "max_conversation_history" not in st.session_state:
    st.session_state.max_conversation_history = 5  # 设置对话历史记录最大条数
if "include_thinking_in_history" not in st.session_state:
    st.session_state.include_thinking_in_history = False  # 是否在对话历史中包含思考过程
if "use_stream" not in st.session_state:
    st.session_state.use_stream = True  # 默认启用流式响应
# LLM配置
if "llm_type" not in st.session_state:
    st.session_state.llm_type = "ollama"  # 默认使用ollama
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = OLLAMA_API_URL
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = OLLAMA_MODEL
if "custom_base_url" not in st.session_state:
    st.session_state.custom_base_url = ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "custom_model" not in st.session_state:
    st.session_state.custom_model = "deepseek-chat"
# 组件加载状态
if "components_loaded" not in st.session_state:
    st.session_state["components_loaded"] = False
# --- End 会话状态初始化 ---

# 初始化组件（在主线程中运行）
@st.cache_resource(ttl=3600)
def load_components():
    """加载并缓存组件，确保只初始化一次"""
    global global_components
    if global_components is None:
        global_components = init_components()
    return global_components

# 应用组件到会话状态
def update_session_with_components():
    """将组件添加到会话状态"""
    components = load_components()
    if components and components.get("components_loaded", False):
        # 更新会话状态中的组件
        for key, value in components.items():
            st.session_state[key] = value
        return True
    return False

# 获取Ollama可用模型列表
def get_ollama_models(api_url: str = "http://localhost:11434") -> List[str]:
    """获取Ollama服务器上可用的模型列表"""
    try:
        import requests
        response = requests.get(f"{api_url}/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        else:
            logger.warning(f"获取Ollama模型列表失败，状态码: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取Ollama模型列表时出错: {e}")
        return []

# 记录prompt和上下文信息
# 全局消息集合用于日志去重
logged_messages = set()

def log_prompt_context(prompt: str, context: List[str], references: List[str], llm_response: Optional[str] = None):
    """记录发送给LLM的prompt内容和上下文信息，并验证输入输出
    
    Args:
        prompt: 发送给LLM的prompt内容
        context: 上下文信息列表
        references: 参考资料列表
        llm_response: LLM的响应内容(可选)
    """
    if 1:
        
        def prompt_logger(message: str):
            if message not in logged_messages:
                prompt_logger.info(message)
                logged_messages.add(message)
        
        # 记录时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        prompt_logger.info(f"===== 交互时间: {timestamp} =====")
        
        # 记录输入验证
        prompt_logger.info("===== 输入验证 =====")
        prompt_logger.info(f"Prompt长度: {len(prompt)} 字符")
        prompt_logger.info(f"上下文数量: {len(context)}")
        prompt_logger.info(f"参考资料数量: {len(references)}")
        
        # 记录详细内容
        prompt_logger.info("===== 实际Prompt内容 =====")
        prompt_logger.info(prompt)
        
        prompt_logger.info("===== 上下文信息 =====")
        for idx, ctx in enumerate(context, 1):
            prompt_logger.info(f"上下文 {idx}({len(ctx)}字符): {ctx[:200]}...")
            
        prompt_logger.info("===== 参考资料 =====")
        for idx, ref in enumerate(references, 1):
            prompt_logger.info(f"参考 {idx}({len(ref)}字符): {ref[:200]}...")
            
        # 记录输出验证(如果有)
        if llm_response:
            prompt_logger.info("===== 输出验证 =====")
            prompt_logger.info(f"响应长度: {len(llm_response)} 字符")
            prompt_logger.info(f"响应内容: {llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")
            
        prompt_logger.info("=" * 40)

# --- 用户认证 ---
def authenticate(username: str, password: str) -> bool:
    """用户认证"""
    # 这里应该从数据库或配置文件中获取用户信息
    users = {
        "admin": bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()),
        "user": bcrypt.hashpw("user123".encode("utf-8"), bcrypt.gensalt()),
    }

    if username in users:
        return bcrypt.checkpw(password.encode("utf-8"), users[username])
    return False

def extract_thinking_process(content: str) -> tuple:
    """从内容中提取思考过程并清理回答

    Args:
        content: 包含思考过程的完整回答内容

    Returns:
        tuple: (清理后的内容, 思考过程内容)，如果没有思考过程，则思考过程为None
    """
    thought_content = None
    cleaned_content = content

    if "<think>" in content and "</think>" in content:
        try:
            # 提取思考过程内容
            start_index = content.find("<think>") + len("<think>")
            end_index = content.find("</think>")
            if start_index > 0 and end_index > start_index:
                thought_content = content[start_index:end_index].strip()

                # 从答案中完全删除思考过程部分，包括标签
                cleaned_content = content[:content.find("<think>")] + content[content.find("</think>") + len("</think>"):].strip()
        except Exception as e:
            logger.error(f"解析思考过程时出错: {e}")

    return cleaned_content, thought_content

# 自定义CSS
def apply_custom_css():
    """应用自定义CSS样式"""
    st.markdown(
        """
    <style>
        /* 页面整体样式 */
        :root {
            --primary-color: #4f6a92;
            --secondary-color: #f5f7fa;
            --text-color: #333333;
            --border-radius: 8px;
            --box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        .main {
            background-color: #fff;
            padding: 1rem;
        }

        /* 标题样式 */
        h1 {
            color: var(--text-color);
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* 聊天消息容器样式 */
        [data-testid="stChatMessageContent"] {
            max-width: 90%;
            padding: 0.8rem 1rem;
            border: none !important;
            box-shadow: var(--box-shadow);
            border-radius: var(--border-radius);
            line-height: 1.5;
        }

        /* 用户消息样式 */
        [data-testid="stChatMessage"][data-testid="user"] [data-testid="stChatMessageContent"] {
            background-color: var(--primary-color) !important;
            color: white !important;
        }

        /* 助手消息样式 */
        [data-testid="stChatMessage"][data-testid="assistant"] [data-testid="stChatMessageContent"] {
            background-color: var(--secondary-color) !important;
            color: var(--text-color) !important;
        }

        /* 扩展器样式 */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: var(--text-color);
            background-color: white;
            border-radius: var(--border-radius);
            padding: 0.7rem 1rem;
            margin-bottom: 0.5rem;
            border: 1px solid #eaeaea;
        }

        /* 扩展器内容样式 */
        .streamlit-expanderContent {
            padding: 10px;
            background-color: #fafafa;
            border-radius: 0 0 var(--border-radius) var(--border-radius);
            border: 1px solid #eaeaea;
            border-top: none;
            margin-bottom: 10px;
        }

        /* 思考过程容器样式 */
        .thinking-container {
            padding: 0.8rem;
            background-color: #f8f9fa;
            border-radius: var(--border-radius);
            border-left: 3px solid var(--primary-color);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

# 应用自定义CSS
apply_custom_css()

# --- 主界面 ---
st.title("🏺 智能陶瓷问答系统")
st.markdown("##### 一器一世界，陶然共古今")

# --- 侧边栏 ---
with st.sidebar:
    st.title("导航与设置")

    # 检查并加载组件
    components_loaded = update_session_with_components()

    # 显示加载状态
    if not components_loaded:
        with st.spinner("系统组件正在加载中..."):
            st.info("系统组件正在加载，某些功能可能暂时不可用。")

            # 显示加载进度占位符
            progress_placeholder = st.empty()
            # 每3秒检查一次是否加载完成
            if "loading_progress" not in st.session_state:
                st.session_state["loading_progress"] = 0

            if st.session_state["loading_progress"] < 100:
                progress_bar = progress_placeholder.progress(st.session_state["loading_progress"])
                st.session_state["loading_progress"] += 20
                if st.session_state["loading_progress"] > 100:
                    st.session_state["loading_progress"] = 100
    else:
        st.success("系统组件已加载完成！")

    if st.session_state.user_role is None:
        st.header("登录")
        login_username = st.text_input("用户名", key="login_user")
        login_password = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录", key="login_button"):
            if authenticate(login_username, login_password):
                st.session_state.user_role = (
                    "admin" if login_username == "admin" else "user"
                )
                st.rerun()  # 重新运行以更新界面
            else:
                st.error("用户名或密码错误")
    else:
        st.success(f"欢迎, {st.session_state.user_role}!")
        if st.button("退出登录"):
            # 清理会话状态
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()  # 重新运行回到登录状态

        # --- 管理员侧边栏 ---
        if st.session_state.user_role == "admin":
            st.markdown("---")
            st.header("📁 文件管理")
            uploaded_files = st.file_uploader(
                "上传文件 (PDF/TXT/JSON/MD等)",
                type=["pdf", "txt", "json", "md", "docx", "csv"],
                accept_multiple_files=True,
                key="admin_uploader",  # 添加唯一key
            )

            if uploaded_files:
                # 确保组件已加载
                if not components_loaded:
                    st.warning("系统组件仍在加载中，请稍候再试...")
                else:
                    for file in uploaded_files:
                        # 计算文件的MD5值
                        import hashlib
                        file_content = file.getbuffer()
                        file_md5 = hashlib.md5(file_content).hexdigest()

                        # 检查是否有相同MD5的文件已存在于缓存中
                        md5_exists = False
                        file_to_replace = None

                        for cache_file in os.listdir(CACHE_DIR):
                            cache_file_path = os.path.join(CACHE_DIR, cache_file)
                            if os.path.isfile(cache_file_path) and not cache_file.startswith("qa_cache"):
                                try:
                                    with open(cache_file_path, "rb") as f:
                                        existing_md5 = hashlib.md5(f.read()).hexdigest()
                                        if existing_md5 == file_md5:
                                            md5_exists = True
                                            break
                                        # 如果文件名相同但MD5不同，标记为需要替换
                                        if cache_file == file.name and existing_md5 != file_md5:
                                            file_to_replace = cache_file_path
                                except Exception as e:
                                    logger.error(f"计算缓存文件MD5时出错: {e}")

                        # 根据检查结果处理文件
                        if md5_exists:
                            st.warning(f"文件 {file.name} 的内容已存在于缓存区中，跳过上传。")
                        else:
                            # 如果存在同名但内容不同的文件，先删除它
                            if file_to_replace and os.path.exists(file_to_replace):
                                try:
                                    os.remove(file_to_replace)
                                    logger.info(f"已删除同名旧文件: {file_to_replace}")
                                except Exception as e:
                                    logger.error(f"删除同名旧文件时出错: {e}")

                            # 保存新文件到缓存
                            cache_file_path = os.path.join(CACHE_DIR, file.name)
                            with open(cache_file_path, "wb") as f:
                                f.write(file_content)
                            st.success(f"文件 {file.name} 已上传到缓存区")

            st.markdown("---")
            st.header("🔄 知识库构建")

            # 1. 处理缓存区文件按钮
            if st.button("处理缓存文件", key="process_cache"):
                # 确保组件已加载
                if not components_loaded:
                    st.warning("系统组件仍在加载中，请稍候再试...")
                else:
                    with st.spinner("正在处理文件..."):
                        processed_count = 0
                        error_count = 0
                        # 修正：排除 qa_cache 目录下的 cache.json 文件
                        cache_files = [
                            f
                            for f in os.listdir(CACHE_DIR)
                            if os.path.isfile(os.path.join(CACHE_DIR, f))
                            and not (
                                f == "cache.json"
                                and os.path.dirname(os.path.join(CACHE_DIR, f))
                                == QA_CACHE_DIR
                            )
                            and not f.startswith("qa_cache")  # 避免意外匹配
                        ]

                        if not cache_files:
                            st.info("缓存区没有待处理的文件。")
                        else:
                            progress_bar = st.progress(0)
                            for i, filename in enumerate(cache_files):
                                try:
                                    # 使用文件管理器处理文件
                                    result = st.session_state["file_manager"].process_file(filename)
                                    st.success(
                                        f"文件 {filename} 已处理并保存为 {os.path.basename(result['file_path'])}"
                                    )
                                    processed_count += 1
                                except Exception as e:
                                    st.error(f"处理文件 {filename} 时出错: {str(e)}")
                                    error_count += 1
                                progress_bar.progress((i + 1) / len(cache_files))
                            st.info(
                                f"处理完成：成功 {processed_count} 个, 失败 {error_count} 个。请点击下方按钮重建索引。"
                            )

            # 2. 重建索引按钮
            if st.button("重建知识库索引", key="rebuild_index"):
                # 确保组件已加载
                if not components_loaded:
                    st.warning("系统组件仍在加载中，请稍候再试...")
                else:
                    with st.spinner("正在读取文件并构建索引和知识图谱..."):
                        try:
                            from utils.file_handlers.file_manager import FileManager
                            from utils.graph.knowledge_graph import KnowledgeGraph
                            from utils.retrieval.vector_store import VectorStore
                            from utils.retrieval.retriever import HybridRetriever

                            # 初始化新的向量存储和知识图谱实例
                            st.write("正在构建知识图谱...")
                            new_knowledge_graph = KnowledgeGraph()

                            # 初始化新的向量存储
                            st.write("正在创建向量存储...")
                            new_vector_store = VectorStore(
                                storage_dir=STORAGE_DIR,
                                embedding_model=EMBEDDINGS_MODEL,
                                base_url=OLLAMA_API_URL,
                                debug_mode=DEBUG_MODE,
                                log_file=LOG_FILE
                            )

                            # 读取所有文本文件
                            all_texts = []
                            all_metadatas = []
                            for root, _, files in os.walk(DATA_DIR):
                                for file in files:
                                    if file.endswith((".txt", ".md", ".json")):
                                        try:
                                            file_path = os.path.join(root, file)
                                            with open(file_path, "r", encoding="utf-8") as f:
                                                content = f.read()
                                                all_texts.append(content)
                                                all_metadatas.append({
                                                    "source": file_path,
                                                    "type": os.path.splitext(file)[1][1:],
                                                    "timestamp": datetime.now().isoformat()
                                                })
                                        except Exception as e:
                                            st.error(f"读取文件 {file} 时出错: {str(e)}")

                            # 创建向量存储
                            if all_texts:
                                new_vector_store.create_from_texts(all_texts, all_metadatas)
                                st.write(f"向量存储创建完成，包含 {len(all_texts)} 个文档。")
                            else:
                                st.warning("没有找到任何文本文件来创建向量存储。")

                            # 创建新的混合检索器
                            new_hybrid_retriever = HybridRetriever(
                                vector_store=new_vector_store,
                                use_reranker=True,
                                rerank_model="BAAI/bge-reranker-base"
                            )

                            # 构建和保存知识图谱
                            new_knowledge_graph.build_from_text("\n\n".join(all_texts))
                            new_knowledge_graph.save_graph(GRAPH_PATH)
                            st.write(f"知识图谱构建完成 ({new_knowledge_graph.graph.number_of_nodes()} 节点, {new_knowledge_graph.graph.number_of_edges()} 边) 并已保存。")

                            # 保存向量存储
                            new_vector_store.save(VECTOR_STORE_FILENAME)
                            st.write(f"向量存储已保存到: {VECTOR_STORE_FILENAME}")

                            # 在Streamlit会话状态中更新实例
                            st.session_state["knowledge_graph_instance"] = new_knowledge_graph
                            st.session_state["hybrid_retriever_instance"] = new_hybrid_retriever
                            st.session_state["vector_store_instance"] = new_vector_store

                            # 调用update_session_with_components以刷新缓存
                            st.rerun()

                            st.success("知识库索引和知识图谱已成功重建！")
                        except Exception as e:
                            st.error(f"重建索引时发生错误: {str(e)}")
                            if DEBUG_MODE:
                                st.code(traceback.format_exc())

            st.header(" 知识库管理 (自定义内容)")
            try:
                # 确保组件已加载
                if components_loaded and "file_manager" in st.session_state:
                    data_files_list = st.session_state["file_manager"].list_files(CUSTOM_FILES_DIR)
                    if data_files_list:
                        st.dataframe(data_files_list)
                        selected_file_data = st.selectbox(
                            "选择文件进行操作",
                            [f["name"] for f in data_files_list],
                            key="data_select",
                        )
                        col1_data, col2_data = st.columns(2)
                        with col1_data:
                            if st.button("删除文件", key="delete_data"):
                                if st.session_state["file_manager"].delete_file(
                                    selected_file_data, CUSTOM_FILES_DIR
                                ):
                                    st.success("文件已删除")
                                    st.rerun()
                                else:
                                    st.error("删除文件失败")
                        with col2_data:
                            new_name_data = st.text_input(
                                "新文件名", key="rename_data_input"
                            )
                            if st.button("重命名文件", key="rename_data"):
                                if new_name_data and st.session_state["file_manager"].rename_file(
                                    selected_file_data, new_name_data, CUSTOM_FILES_DIR
                                ):
                                    st.success("文件已重命名")
                                    st.rerun()
                                else:
                                    st.error("重命名失败或未输入新文件名")
                    else:
                        st.info("目录为空。")
                else:
                    st.warning("组件正在加载中，请稍候再试...")
            except FileNotFoundError:
                st.info("目录不存在或为空。")
            except Exception as e:
                st.error(f"加载目录文件列表时出错: {e}")

            st.markdown("---")
            st.header("⚙️ 系统配置")

            # LLM类型选择
            st.session_state.llm_type = st.radio(
                "选择LLM类型",
                options=["ollama", "自定义LLM"],
                horizontal=True,
                key="llm_type_radio",
            )

            # 根据选择的LLM类型显示不同的配置选项
            if st.session_state.llm_type == "ollama":
                st.session_state.ollama_url = st.text_input(
                    "Ollama API URL",
                    value=st.session_state.ollama_url,
                    key="ollama_url_input",
                )

                # 获取Ollama可用模型列表
                ollama_models = get_ollama_models(st.session_state.ollama_url)

                if ollama_models:
                    st.session_state.ollama_model = st.selectbox(
                        "Ollama 模型",
                        options=ollama_models,
                        index=(
                            ollama_models.index(st.session_state.ollama_model)
                            if st.session_state.ollama_model in ollama_models
                            else 0
                        ),
                        key="ollama_model_select",
                    )
                else:
                    st.warning("无法获取Ollama模型列表，请检查Ollama服务是否正常运行")
                
                st.session_state.ollama_model = st.text_input(
                    "Ollama 模型",
                    value=st.session_state.ollama_model,
                    key="ollama_model_input",
                )
                
                # 添加流式响应选项
                if "use_stream" not in st.session_state:
                    st.session_state.use_stream = True
                st.session_state.use_stream = st.checkbox(
                    "启用流式响应",
                    value=st.session_state.use_stream,
                    help="启用流式响应可以实时显示生成过程，提升用户体验",
                    key="ollama_stream_checkbox",
                )
            else:
                # 自定义LLM配置
                st.session_state.custom_base_url = st.text_input(
                    "自定义LLM Base URL (如 https://api.deepseek.com)",
                    value=st.session_state.custom_base_url,
                    key="custom_base_url_input",
                )
                st.session_state.openai_key = st.text_input(
                    "API Key",
                    value=st.session_state.openai_key,
                    type="password",
                    key="openai_key_input",
                )
                st.session_state.custom_model = st.text_input(
                    "模型名称 (如 deepseek-chat)",
                    value=st.session_state.custom_model,
                    key="custom_model_input",
                )

                # 添加流式响应选项
                if "use_stream" not in st.session_state:
                    st.session_state.use_stream = False
                st.session_state.use_stream = st.checkbox(
                    "启用流式响应",
                    value=st.session_state.use_stream,
                    help="启用流式响应可能会提高响应速度，但某些API可能不支持",
                    key="use_stream_checkbox",
                )

            # 参数调整
            st.session_state.temperature = st.slider(
                "LLM 温度", 0.0, 1.0, st.session_state.temperature, key="temp_slider",
                help="设置LLM的温度，越高越随机但可能更准确"
            )
            st.session_state.max_contexts = st.slider(
                "最大上下文数量 (检索)",
                1,
                10,
                st.session_state.max_contexts,
                key="context_slider",
                help="设置检索时使用的上下文数量，越高越精确但会消耗更多资源"
            )
            st.session_state.max_conversation_history = st.slider(
                "保留对话轮数",
                1,
                10,
                st.session_state.max_conversation_history,
                key="history_slider",
                help="设置保留多少轮对话历史，每轮包含一个用户问题和一个助手回答"
            )
            
            st.session_state.include_thinking_in_history = st.checkbox(
                "在历史中包含思考过程",
                value=st.session_state.include_thinking_in_history,
                help="启用后，有助于保持推理连贯性, 但会增加响应时间，消耗更多资源",
                key="include_thinking_checkbox"
            )

            st.markdown("---")
            st.header("🗄️ 问答缓存管理")
            try:
                # 确保组件已加载
                if components_loaded and "cache_manager" in st.session_state:
                    cache_stats = st.session_state["cache_manager"].get_stats()
                    st.write(f"缓存大小: {cache_stats['size']}/{cache_stats['max_size']}")
                    st.write(f"缓存过期时间 (TTL): {cache_stats['ttl']} 秒")

                    if st.button("清空问答缓存", key="clear_qa_cache"):
                        st.session_state["cache_manager"].clear()
                        st.success("问答缓存已清空")
                else:
                    st.warning("组件正在加载中，请稍候再试...")
            except Exception as e:
                st.error(f"获取缓存状态时出错: {e}")
        # --- End 管理员侧边栏 ---

# --- 显示主界面内容 ---
if st.session_state.user_role is None:
    st.info("👈 请在侧边栏登录以开始提问。")
else:
    # --- 聊天界面 ---
    # 添加控制按钮（在侧边栏）
    with st.sidebar:
        st.markdown("---")
        st.header("💬 聊天设置")

        # 使用列布局让控制按钮更美观
        show_context = st.toggle(
            "显示检索上下文", value=False, key="show_context_toggle", help="显示检索到的相关文档内容"
        )
        show_knowledge_graph = st.toggle(
            "显示知识图谱", value=False, key="show_kg_toggle", help="显示相关的知识图谱信息"
        )
        
        # 添加清除历史对话按钮
        st.markdown("---")
        if st.button("🗑️ 清除历史对话", key="clear_history_button", type="primary"):
            # 清除消息历史
            st.session_state.messages = []
            # 清除问答缓存
            if "cache_manager" in st.session_state and st.session_state["cache_manager"]:
                st.session_state["cache_manager"].clear()
            st.success("历史对话已清除！")
            st.rerun()

    # 显示历史消息
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👤"):
                st.markdown(message["content"])
        else:
            # 提取助手消息中的思考过程
            content, thought_content = extract_thinking_process(message["content"])
            
            # 显示助手消息 - 使用当前选择的LLM类型的头像
            avatar_icon = "🤖"  
            
            # 显示助手消息
            with st.chat_message("assistant", avatar=avatar_icon):
                # 首先显示思考过程（如果有）
                if thought_content:
                    with st.expander("🧠 思考过程", expanded=False):
                        st.markdown(f"<div class='thinking-container'>{thought_content}</div>", unsafe_allow_html=True)
                
                # 然后显示答案
                st.markdown(content)

    # 处理用户输入
    if prompt := st.chat_input("请输入您的问题，例如：什么是景德镇陶瓷？"):
        # 添加用户消息到历史记录前，确保历史记录不超过最大限制
        # 每组对话包含用户问题和助手回答，因此检查当前消息数量是否超过 (max_history*2)-1
        max_history_items = st.session_state.max_conversation_history * 2
        if len(st.session_state.messages) >= max_history_items:
            # 移除最早的一组对话（一问一答）
            st.session_state.messages = st.session_state.messages[2:]

        # 添加用户消息到历史记录
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 检查是否所有组件已加载
        components_loaded = update_session_with_components()
        if not components_loaded:
            with st.chat_message("assistant", avatar="🏺"):
                st.warning("系统组件正在加载中，请稍后再试...")
            st.session_state.messages.append(
                {"role": "assistant", "content": "系统组件正在加载中，请稍后再试..."}
            )
        else:
            # 检查缓存
            cached_result = st.session_state["cache_manager"].get(prompt)
            if cached_result:
                with st.chat_message("assistant", avatar="🏺"):
                    # 提取并显示思考过程（如果有）
                    content, thought_content = extract_thinking_process(cached_result["answer"])

                    # 首先显示思考过程（如果有）
                    if thought_content:
                        with st.expander("🧠 思考过程", expanded=False):
                            st.markdown(f"<div class='thinking-container'>{thought_content}</div>", unsafe_allow_html=True)

                    # 然后显示答案
                    st.markdown(content)

                # 将助手回复添加到消息历史和缓存（包含思考过程，使用标签包装）
                full_message = content
                if thought_content:
                    full_message = f"<think>{thought_content}</think>\n\n{content}"
                    
                st.session_state["cache_manager"].set(prompt, {"answer": full_message})
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_message}
                )
            else:
                # 生成回答
                with st.spinner("思考中..."):
                    try:
                        import requests
                        # 使用会话状态中的检索器
                        current_retriever = st.session_state["hybrid_retriever_instance"]
                        current_graph = st.session_state["knowledge_graph_instance"]

                        # 1. 检索相关文档
                        docs = current_retriever.retrieve(
                            prompt, k=st.session_state.max_contexts
                        )

                        # 显示检索到的文档（如果开启了显示上下文选项）
                        if show_context and docs:
                            with st.expander("📚 相关资料", expanded=False):
                                for i, doc in enumerate(docs):
                                    # 获取文件路径，提取文件名并移除扩展名
                                    source_path = doc.get('metadata', {}).get('source', '未知来源')
                                    file_name = os.path.basename(source_path)
                                    file_name_without_ext = os.path.splitext(file_name)[0]
                                    st.markdown(
                                        f"**资料 {i+1}** - {file_name_without_ext}"
                                    )
                                    st.markdown(doc.get("content", "内容为空"))

                        # 2. 查询知识图谱
                        graph_context = ""
                        graph_results = []
                        if (
                            current_graph
                            and hasattr(current_graph, "graph")
                            and current_graph.graph.number_of_nodes() > 0
                        ):
                            try:
                                # 限制只返回前top_k个最相关的实体
                                graph_results = current_graph.query(prompt, top_k=20)
                                if graph_results:
                                    graph_context_list = []
                                    for result in graph_results:
                                        # 只取前top_k个相关实体，减少token使用
                                        related = ", ".join(
                                            result.get("related_entities", [])[:10]
                                        )
                                        graph_context_list.append(
                                            f"「知识图谱相关: {result['entity']} 关联词: {related}」"
                                        )
                                    graph_context = "\n".join(graph_context_list)

                                    # 显示知识图谱结果（如果开启了显示知识图谱选项）
                                    if show_knowledge_graph and graph_context:
                                        with st.expander(
                                            "🔍 知识图谱", expanded=False
                                        ):
                                            for result in graph_results:
                                                st.markdown(
                                                    f"**实体:** {result['entity']}"
                                                )
                                                if (
                                                    "related_entities" in result
                                                    and result["related_entities"]
                                                ):
                                                    st.markdown(
                                                        f"**关联实体:** {', '.join(result['related_entities'])}"
                                                    )
                            except Exception as e:
                                st.warning(f"查询知识图谱时出错: {e}")

                        # 3. 构建提示
                        # 构建上下文信息
                        context_parts = []

                        # 添加检索到的文档
                        if docs:
                            doc_texts = []
                            for i, doc in enumerate(docs):                                
                                content = doc.get("content", "")
                                # 提取文件名并移除扩展名
                                # source = doc.get("metadata", {}).get(
                                #     "source", f"来源{i+1}"
                                # )
                                # doc_texts.append(f"[{source}]: {content}")

                                doc_texts.append(f"参考资料 {i+1}: {content}")
                            context_parts.append("\n".join(doc_texts))

                        # 添加知识图谱信息
                        if graph_context:
                            context_parts.append(f"知识图谱信息:\n{graph_context}")

                        # 合并上下文信息
                        context_str = (
                            "\n\n".join(context_parts)
                            if context_parts
                            else "无可用上下文信息"
                        )

                        # 构建历史对话上下文
                        chat_history = ""
                        if len(st.session_state.messages) > 1:
                            # 获取最近的对话历史（不包括当前问题）
                            history_messages = st.session_state.messages[:-1]
                            
                            # 限制历史消息数量
                            max_pairs = st.session_state.max_conversation_history - 1
                            if len(history_messages) > max_pairs * 2:
                                history_messages = history_messages[-(max_pairs * 2):]
                            
                            # 格式化对话历史 - 优化格式更简洁清晰
                            formatted_history = []
                            for i in range(0, len(history_messages), 2):
                                if i+1 < len(history_messages):
                                    user_msg = history_messages[i]["content"]
                                    assistant_msg = history_messages[i+1]["content"]
                                    
                                    # 提取思考过程和答案内容
                                    answer_content, thought_content = extract_thinking_process(assistant_msg)
                                    
                                    # 根据配置决定是否在历史对话中包含思考过程
                                    include_thinking_in_history = st.session_state.include_thinking_in_history
                                    
                                    # 优化历史格式更简洁
                                    if include_thinking_in_history and thought_content:
                                        assistant_formatted = f"思考：{thought_content}\n答案：{answer_content}"
                                    else:
                                        assistant_formatted = answer_content
                                        
                                    formatted_history.append(f"用户：{user_msg}\n助手：{assistant_formatted}")
                            
                            chat_history = "\n\n".join(formatted_history)

                        # 构建系统提示词 - 优化提示词减少对参考资料的过度关注
                        system_prompt = """你是一个专注于陶瓷领域的智能助手，具有以下特点：
1. 专业性：精通陶瓷知识，对陶瓷文化有深刻理解
2. 上下文理解：能准确理解和利用对话历史，保持对话连贯性
3. 思维透明：提供清晰的推理过程，解释如何得出结论
4. 知识整合：结合检索到的文档和知识图谱信息提供全面答案
5. 对于基本的常识问题，能提供基本信息,对于专业名称或者概念，参考资料给出答案

在回答时请遵循以下原则：
1. 保持答案简洁、清晰、直接回应用户问题
2. 优先使用提供的上下文信息，但不要生硬复制或逐条总结参考资料
3. 避免过度解释参考资料来源，除非用户明确询问信息来源
4. 保持自然的对话风格，不要过度学术化
5. 不要胡编乱造资料,确保回答有据可依，但以用户问题为中心
6. 若用户表达非陶瓷相关问题，可礼貌致谢并引导至陶瓷话题, 并引导用户继续提问"""

                        # 构建用户提示词 - 优化指令更加明确
                        user_prompt = f"""请回答下面的问题，简洁明了地直接解答用户的疑问：

__当前问题：{prompt}__

你可以参考以下信息,但是不要将这些消息误认为问题，这些只是资料：

【历史对话】
{chat_history if chat_history else "这是对话的开始"}

【参考资料】
{context_str}

请注意：
- 如果历史对话和参考资料都没有提供相关信息，直接回答"我不确定"或"我无法回答"，不要编造答案，引导用户继续提问获得更多信息
- 只使用与问题相关的信息，无关内容可忽略
"""

                        # 构建消息列表
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        # print(messages)
                        # 4. 调用LLM - 统一处理函数
                        # 创建流式响应
                        full_response = ""

                        try:
                            # 创建消息容器 - 根据LLM类型选择不同的头像
                            avatar_icon = "🏺" 
                            with st.chat_message("assistant", avatar=avatar_icon):
                                message_placeholder = st.empty()

                                # 根据选择的LLM类型进行不同的处理
                                if st.session_state.llm_type == "ollama":
                                    # 使用Ollama API
                                    try:
                                        # 确定是否使用流式响应
                                        use_stream = st.session_state.use_stream if "use_stream" in st.session_state else True
                                        
                                        # 记录发送给Ollama的prompt内容
                                        logger.info(f"发送给Ollama的prompt内容:\n{system_prompt}")
                                        logger.info(f"Ollama请求参数: model={st.session_state.ollama_model}, temperature={st.session_state.temperature}")
                                        
                                        # 构建完整提示，包含系统提示、用户问题和上下文
                                        complete_prompt = f"{system_prompt}\n\n{user_prompt}"
                                        
                                        # 更详细的日志记录
                                        if DEBUG_MODE:
                                            logger.debug(f"发送给Ollama的完整提示:\n{complete_prompt}")
                                            # 记录输入到Prompt日志
                                            log_prompt_context(
                                                complete_prompt, 
                                                [context_str] if context_str else [], 
                                                [], 
                                                None
                                            )
                                        
                                        response = requests.post(
                                            f"{st.session_state.ollama_url}/api/generate",
                                            json={
                                                "model": st.session_state.ollama_model,
                                                "prompt": complete_prompt,  # 使用完整提示
                                                "stream": use_stream,
                                                "options": {
                                                    "temperature": st.session_state.temperature,
                                                    "num_ctx": 8192
                                                }
                                            },
                                            stream=use_stream,
                                            timeout=60
                                        )
                                        response.raise_for_status()

                                        if use_stream:
                                            # 处理流式响应
                                            for line in response.iter_lines():
                                                if line:
                                                    data = json.loads(line.decode())
                                                    token = data.get("response", "")
                                                    full_response += token
                                                    message_placeholder.markdown(full_response + "▌")
                                                    if data.get("done", False):
                                                        logger.info(f"Ollama流式响应完成: {full_response}")
                                                        break
                                        else:
                                            # 处理非流式响应
                                            response_data = response.json()
                                            full_response = response_data.get("response", "抱歉，我无法生成回答。请检查Ollama服务是否运行以及模型是否正确。")
                                            logger.info(f"Ollama完整响应: {full_response}")
                                    except requests.exceptions.RequestException as e:
                                        error_msg = "连接Ollama服务器失败，请检查服务是否运行"
                                        logger.error(f"Ollama API错误: {str(e)}")
                                        raise ValueError(error_msg)

                                else:  # 使用自定义LLM (OpenAI兼容接口)
                                    # 检查必要的配置
                                    if not st.session_state.custom_base_url or not st.session_state.openai_key:
                                        raise ValueError("请在系统配置中设置完整的自定义LLM信息（Base URL和API Key）")

                                    try:
                                        # 确保URL格式正确
                                        base_url = st.session_state.custom_base_url.strip()
                                        
                                        # 移除可能重复的协议前缀
                                        base_url = base_url.replace("https://https://", "https://")
                                        base_url = base_url.replace("http://http://", "http://")
                                        
                                        # 如果没有协议前缀，添加https://
                                        if not base_url.startswith(('http://', 'https://')):
                                            base_url = f"https://{base_url}"
                                        
                                        # 移除URL末尾的斜杠
                                        base_url = base_url.rstrip('/')
                                        
                                        logger.info(f"正在连接自定义LLM API: {base_url}")
                                        
                                        # 初始化OpenAI客户端
                                        client = OpenAI(
                                            api_key=st.session_state.openai_key,
                                            base_url=base_url,
                                            timeout=60.0,
                                            max_retries=2  # 添加重试次数
                                        )

                                        # 测试API连接
                                        try:
                                            # 尝试一个轻量级的API调用来测试连接
                                            test_response = client.models.list()
                                            logger.info("API连接测试成功")
                                        except Exception as test_e:
                                            logger.error(f"API连接测试失败: {str(test_e)}")
                                            if "auth" in str(test_e).lower():
                                                raise ValueError("API认证失败，请检查API密钥是否正确")
                                            elif any(err in str(test_e).lower() for err in ["connection", "timeout", "connect"]):
                                                raise ValueError(f"无法连接到服务器 {base_url}，请检查:\n1. 服务器地址是否正确\n2. 网络连接是否正常\n3. 服务器是否在线")
                                            else:
                                                raise ValueError(f"API连接测试失败: {str(test_e)}")

                                        # 构建消息列表
                                        messages = [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt}
                                        ]

                                        # 确定是否使用流式响应
                                        use_stream = st.session_state.use_stream if "use_stream" in st.session_state else False

                                        if use_stream:
                                            # 流式响应处理
                                            stream = client.chat.completions.create(
                                                model=st.session_state.custom_model,
                                                messages=messages,
                                                temperature=st.session_state.temperature,
                                                stream=True
                                            )

                                            # 用于累积思考过程和最终答案
                                            reasoning_content = ""
                                            final_content = ""

                                            for chunk in stream:
                                                if hasattr(chunk.choices[0], 'delta'):
                                                    delta = chunk.choices[0].delta
                                                    # 检查是否有reasoning_content
                                                    if hasattr(delta, 'reasoning_content'):
                                                        reasoning = delta.reasoning_content
                                                        if reasoning:
                                                            reasoning_content += reasoning
                                                    # 检查普通content
                                                    if hasattr(delta, 'content'):
                                                        content = delta.content
                                                        if content:
                                                            final_content += content
                                                            full_response = f"<think>{reasoning_content}</think>\n\n{final_content}"
                                                            message_placeholder.markdown(full_response + "▌")

                                            # 确保最终响应包含思考过程
                                            if reasoning_content:
                                                full_response = f"<think>{reasoning_content}</think>\n\n{final_content}"
                                            else:
                                                full_response = final_content
                                        else:
                                            # 非流式响应
                                            response = client.chat.completions.create(
                                                model=st.session_state.custom_model,
                                                messages=messages,
                                                temperature=st.session_state.temperature,
                                                stream=False
                                            )

                                            # 提取思考过程和最终答案
                                            if hasattr(response.choices[0].message, 'reasoning_content'):
                                                reasoning = response.choices[0].message.reasoning_content
                                                content = response.choices[0].message.content
                                                full_response = f"<think>{reasoning}</think>\n\n{content}"
                                            else:
                                                # 如果没有reasoning_content，就只使用普通content
                                                full_response = response.choices[0].message.content

                                    except Exception as api_error:
                                        error_msg = str(api_error)
                                        logger.error(f"自定义LLM API错误: {error_msg}")
                                        
                                        if isinstance(api_error, ValueError):
                                            # 直接传递ValueError的错误信息
                                            raise ValueError(error_msg)
                                        elif "auth" in error_msg.lower() or "key" in error_msg.lower() or "token" in error_msg.lower():
                                            raise ValueError("API认证失败，请检查API密钥是否正确")
                                        elif any(err in error_msg.lower() for err in ["connection", "timeout", "connect"]):
                                            raise ValueError(f"连接服务器失败 ({base_url})，请检查:\n1. 服务器地址是否正确\n2. 网络连接是否正常\n3. 服务器是否在线")
                                        else:
                                            raise ValueError(f"调用API时发生错误: {error_msg}")

                                # 清空流式响应占位符
                                message_placeholder.empty()

                                # 解析思考过程并清理答案
                                answer, thought_content = extract_thinking_process(full_response)

                                # 首先显示思考过程（如果有）
                                if thought_content:
                                    with st.expander("🧠 思考过程", expanded=False):
                                        st.markdown(f"<div class='thinking-container'>{thought_content}</div>", unsafe_allow_html=True)

                                # 然后在消息容器内显示最终答案（不含思考过程及标签）
                                st.markdown(answer)

                            # 将助手回复添加到消息历史和缓存（包含思考过程，使用标签包装）
                            full_message = answer
                            if thought_content:
                                full_message = f"<think>{thought_content}</think>\n\n{answer}"
                                
                            st.session_state["cache_manager"].set(prompt, {"answer": full_message})
                            st.session_state.messages.append(
                                {"role": "assistant", "content": full_message}
                            )

                        except ValueError as ve:
                            # 处理参数错误
                            with st.chat_message("assistant", avatar="🏺"):
                                st.error(f"参数错误: {str(ve)}")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"配置错误: {ve}"}
                            )
                        except Exception as e:
                            # 处理其他错误
                            with st.chat_message("assistant", avatar="🏺"):
                                error_msg = f"生成回答时发生错误: {str(e)}"
                                if "auth" in str(e).lower() or "key" in str(e).lower() or "token" in str(e).lower():
                                    error_msg = "认证失败，请检查API密钥是否正确"
                                elif "connect" in str(e).lower() or "timeout" in str(e).lower():
                                    error_msg = "连接服务器失败，请检查网络或服务器地址"
                                st.error(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"处理时发生错误: {e}"}
                            )
                    except Exception as e:
                        # 这里捕获上层其他可能的错误，比如检索器错误等
                        with st.chat_message("assistant", avatar="🏺"):
                            st.error(f"处理请求时发生错误: {str(e)}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"处理请求时发生错误: {e}"}
                        )
    # --- End 聊天界面 ---
