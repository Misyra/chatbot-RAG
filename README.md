# 陶瓷知识问答机器人

## 项目概述

![架构图](docs/images/architecture.png)

## 目录结构

```
ceramic-rag-chatbot/
├── app.py                # 主应用入口
├── data/                 # 知识库文档
│   ├── custom_files/     # 用户上传文件
│   └── default_data/     # 默认知识库
├── utils/                # 工具模块
│   ├── file_handlers/    # 文件处理器
│   └── retrieval/        # 检索模块
├── storage/              # 向量存储
├── cache/                # 缓存数据
├── logs/                 # 日志文件
└── requirements.txt      # 依赖列表
```

这是一个基于 Streamlit 构建的陶瓷知识问答机器人，使用 LangChain 框架和 FAISS 向量数据库实现 RAG(检索增强生成)功能。项目主要功能包括：

- 陶瓷专业术语和知识的检索
- 自然语言问答交互
- 支持多种文档格式(PDF、Word、Markdown 等)
- 知识库管理和更新

## 技术栈

- 前端: Streamlit
- 后端: Python
- 核心库: LangChain, FAISS, Sentence-Transformers
- 其他: PyMuPDF, python-docx, pypdf 等文档处理库

## 环境要求

- Python 3.10+
- CUDA 11.8 (如需 GPU 加速)
- 至少 8GB 内存
- 依赖包(详见 requirements.txt)

### 开发环境配置

1. 安装 CUDA 工具包(可选)
2. 创建 Python 虚拟环境
3. 安装依赖：`pip install -r requirements.txt`
4. 下载 rerank 模型：`python download_rerank.py`

## 安装指南

1. 克隆项目仓库

```bash
git clone <仓库地址>
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

1. 配置环境变量(复制.env.example 为.env 并填写必要配置)
2. 启动应用

```bash
streamlit run app.py
```

3. 访问 `http://localhost:8501`

## 部署指南

### 性能优化建议

1. 启用 GPU 加速：确保安装正确版本的 CUDA 和 cuDNN
2. 向量数据库优化：定期重建 FAISS 索引以提高检索效率
3. 缓存策略：实现问答结果缓存减少重复计算
4. 批处理：对大量文档处理时使用批处理模式

### 常见问题解答

Q: 如何扩展知识库？
A: 通过上传文档功能添加 PDF/Word 等格式文件

Q: 如何提高问答准确性？
A: 确保知识库文档质量，定期更新索引

Q: 支持哪些语言？
A: 目前主要支持中文问答，未来计划增加多语言支持

1. 生产环境推荐使用 Docker 部署

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

2. 构建并运行容器

```bash
docker build -t ceramic-chatbot .
docker run -p 8501:8501 ceramic-chatbot
```

## 功能说明

1. **知识问答**：输入问题获取陶瓷专业知识的回答
2. **文档上传**：支持上传 PDF、Word 等文档扩展知识库
3. **历史记录**：保存问答历史便于回顾
4. **知识管理**：后台可管理知识库文档


