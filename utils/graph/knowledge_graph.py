import networkx as nx
import re
from typing import List, Dict, Set, Tuple
import jieba
import jieba.posseg as pseg
import math
from collections import Counter

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_types = {
            'n': '名词',
            'nr': '人名',
            'ns': '地名',
            'nt': '机构名',
            'nz': '其他专名',
            'vn': '动名词',
            'ceramic_type': '陶瓷种类',
            'ceramic_tech': '陶瓷工艺',
            'ceramic_material': '陶瓷原料',
            'ceramic_style': '陶瓷风格',
            'ceramic_period': '陶瓷时期'
        }
        # 添加实体共现统计
        self.entity_cooccurrence = Counter()
        # 添加窗口大小参数
        self.window_size = 5
        # 添加最小共现阈值
        self.min_cooccurrence = 2
        # 陶瓷领域特定关系类型
        self.relation_types = {
            'material_relation': '原料关系',
            'tech_relation': '工艺关系',
            'style_relation': '风格关系',
            'period_relation': '时期关系'
        }
        
    def build_from_text(self, text: str) -> None:
        """从文本构建知识图谱，使用共现关系和语义相关性"""
        # 使用jieba进行分词和词性标注
        words_with_pos = list(pseg.cut(text))
        
        # 提取所有实体
        entities = []
        for word, flag in words_with_pos:
            if flag in self.entity_types and len(word) > 1:  # 过滤单字实体，减少噪音
                entities.append((word, flag))
                # 添加实体节点
                if word not in self.graph:
                    self.graph.add_node(word, type=self.entity_types[flag])
        
        # 计算实体共现关系
        for i, (entity1, _) in enumerate(entities):
            # 使用滑动窗口计算共现
            window_end = min(i + self.window_size, len(entities))
            for j in range(i + 1, window_end):
                entity2 = entities[j][0]
                if entity1 != entity2:
                    self.entity_cooccurrence[(entity1, entity2)] += 1
                    self.entity_cooccurrence[(entity2, entity1)] += 1
        
        # 基于共现次数添加边
        for (entity1, entity2), count in self.entity_cooccurrence.items():
            if count >= self.min_cooccurrence and entity1 in self.graph and entity2 in self.graph:
                # 如果边已存在，更新权重；否则创建新边
                if self.graph.has_edge(entity1, entity2):
                    self.graph[entity1][entity2]['weight'] += count
                else:
                    self.graph.add_edge(entity1, entity2, weight=count)
    
    def query(self, query: str, top_k: int = 10) -> List[Dict]:
        """查询知识图谱，返回最相关的前top_k个实体"""
        # 提取查询中的实体
        words = pseg.cut(query)
        query_entities = {word for word, flag in words if flag in self.entity_types}
        
        results = []
        for entity in query_entities:
            if entity in self.graph:
                # 获取直接相连的实体及其权重
                neighbors_with_weights = []
                for neighbor in self.graph.neighbors(entity):
                    weight = self.graph[entity][neighbor].get('weight', 1)
                    neighbors_with_weights.append((neighbor, weight))
                
                # 按权重排序并取前top_k个
                sorted_neighbors = sorted(neighbors_with_weights, 
                                         key=lambda x: x[1], 
                                         reverse=True)[:top_k]
                
                if sorted_neighbors:
                    results.append({
                        "entity": entity,
                        "related_entities": [n[0] for n in sorted_neighbors]
                    })
                    
        return results
        
    def get_entity_info(self, entity: str) -> Dict:
        """获取实体的详细信息"""
        if entity not in self.graph:
            return {}
            
        neighbors = list(self.graph.neighbors(entity))
        return {
            "entity": entity,
            "degree": self.graph.degree(entity),
            "neighbors": neighbors,
            "neighbor_count": len(neighbors)
        }
        
    def get_graph_stats(self) -> Dict:
        """获取图谱统计信息"""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
        
    def save_graph(self, filepath: str) -> None:
        """保存知识图谱"""
        try:
            # 适应新版本和旧版本的networkx
            if hasattr(nx, 'write_gpickle'):
                nx.write_gpickle(self.graph, filepath)
            else:
                # 新版本的networkx中使用nx.readwrite模块
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self.graph, f)
            print(f"图谱已成功保存到: {filepath}")
        except Exception as e:
            print(f"保存图谱时出错: {e}")
            
    def load_graph(self, filepath: str) -> None:
        """加载知识图谱"""
        try:
            # 适应新版本和旧版本的networkx
            if hasattr(nx, 'read_gpickle'):
                self.graph = nx.read_gpickle(filepath)
            else:
                # 新版本的networkx中使用nx.readwrite模块
                import pickle
                with open(filepath, 'rb') as f:
                    self.graph = pickle.load(f)
            print(f"图谱已成功加载: {filepath}")
        except Exception as e:
            print(f"加载图谱时出错: {e}")
            raise