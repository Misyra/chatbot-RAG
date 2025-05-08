import os
# 设置环境变量 HF_ENDPOINT
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer("BAAI/bge-reranker-base")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
# 生成嵌入向量
embeddings = model.encode(sentences)

# 计算相似度
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]