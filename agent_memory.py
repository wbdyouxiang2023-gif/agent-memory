"""
Agent Memory System — 轻量级 AI Agent 语义记忆引擎

让 AI Agent 真正拥有"记忆"：对话自动分块、语义检索、重要性评分、记忆巩固。

核心特性：
- 纯 Python 标准库，零外部依赖
- 三层记忆架构（工作记忆 / 短期记忆 / 长期记忆）
- TF-IDF 语义检索（无需 API）
- 自动记忆巩固与压缩
- SQLite 持久化

用法:
    from agent_memory import AgentMemory

    mem = AgentMemory()
    mem.add_conversation("用户问了什么", "助手回答了什么")
    results = mem.retrieve("和刚才相关的内容")
"""

import sqlite3
import json
import math
import re
import os
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from enum import Enum


class MemoryLayer(Enum):
    """记忆分层"""
    WORKING = "working"    # 当前会话，随时可访问
    SHORT_TERM = "short"   # 最近 7 天，原始对话
    LONG_TERM = "long"     # 已巩固，摘要化


@dataclass
class MemoryEntry:
    """单条记忆"""
    id: str
    content: str
    summary: str = ""
    layer: str = "short"
    importance: float = 0.5
    timestamp: str = ""
    keywords: List[str] = field(default_factory=list)
    vector: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TextProcessor:
    """文本处理：分词、TF-IDF 向量化"""

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.doc_freq: Dict[str, int] = {}
        self.total_docs: int = 0
        self._stop_words = {
            '的', '了', '是', '在', '我', '你', '他', '她', '它', '这', '那',
            '有', '没', '就', '都', '也', '会', '要', '可以', '能', '把', '被',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
            'and', 'or', 'but', 'not', 'this', 'that', 'it', 'as',
        }

    def tokenize(self, text: str) -> List[str]:
        """分词：中文按字/词，英文按词"""
        # 中文按字符级分词（简单有效）
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        # 英文按单词
        english = re.findall(r'[a-zA-Z]+', text)
        # 数字
        numbers = re.findall(r'\d+', text)

        tokens = []
        for c in chinese:
            for i in range(len(c) - 1):
                tokens.append(c[i:i+2])  # bigram
            if len(c) <= 2:
                tokens.append(c)
        tokens.extend(english)
        tokens.extend(numbers)

        return [t.lower() for t in tokens if t.lower() not in self._stop_words and len(t) > 1]

    def fit(self, documents: List[str]):
        """构建词汇表和文档频率"""
        self.total_docs = len(documents)
        self.doc_freq = {}
        self.vocabulary = {}

        all_tokens = set()
        for doc in documents:
            tokens = set(self.tokenize(doc))
            all_tokens.update(tokens)
            for t in tokens:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.vocabulary = {t: i for i, t in enumerate(sorted(all_tokens))}

    def transform(self, text: str) -> Dict[str, float]:
        """将文本转为 TF-IDF 向量（字典表示，稀疏）"""
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        vec = {}

        for token, count in tf.items():
            if token not in self.vocabulary:
                continue
            # TF
            tf_val = count / len(tokens) if tokens else 0
            # IDF
            df = self.doc_freq.get(token, 1)
            idf_val = math.log(self.total_docs / (df + 1)) + 1
            vec[token] = tf_val * idf_val

        # 归一化
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}

        return vec

    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """余弦相似度"""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        dot = sum(vec1[k] * vec2[k] for k in common)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class AgentMemory:
    """AI Agent 语义记忆引擎"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.expanduser("~"), ".agent_memory.db"
            )
        self.db_path = db_path
        self.processor = TextProcessor()
        self._working_memory: List[MemoryEntry] = []
        self._entries: List[MemoryEntry] = []
        self._conn = None
        self._init_db()
        self._load_from_db()
        self._rebuild_index()

    def _get_conn(self):
        """获取数据库连接（:memory: 时保持单例）"""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ─── 数据库 ───

    def _load_from_db(self):
        """从 SQLite 加载记忆"""
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT * FROM memories ORDER BY timestamp")
            for row in cursor:
                entry = MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    summary=row["summary"] or "",
                    layer=row["layer"],
                    importance=row["importance"],
                    timestamp=row["timestamp"],
                    keywords=json.loads(row["keywords"]) if row["keywords"] else [],
                    vector=json.loads(row["vector"]) if row["vector"] else {},
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                self._entries.append(entry)
        except sqlite3.OperationalError:
            pass  # 表不存在
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def _save_to_db(self, entry: MemoryEntry):
        """保存单条记忆到 SQLite"""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, summary, layer, importance, timestamp, keywords, vector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.content, entry.summary, entry.layer,
                entry.importance, entry.timestamp,
                json.dumps(entry.keywords), json.dumps(entry.vector),
                json.dumps(entry.metadata),
            ))
            conn.commit()
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    layer TEXT DEFAULT 'short',
                    importance REAL DEFAULT 0.5,
                    timestamp TEXT NOT NULL,
                    keywords TEXT DEFAULT '[]',
                    vector TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_layer ON memories(layer)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
            """)
            conn.commit()
        finally:
            if self.db_path != ":memory:":
                conn.close()

    # ─── 索引重建 ───

    def _rebuild_index(self):
        """重建 TF-IDF 索引"""
        all_content = [e.content for e in self._entries]
        if all_content:
            self.processor.fit(all_content)
        # 为所有条目生成向量
        for entry in self._entries:
            if not entry.vector:
                entry.vector = self.processor.transform(entry.content)

    # ─── 核心 API ───

    def add_conversation(self, user_msg: str, assistant_msg: str,
                          metadata: Optional[Dict] = None):
        """添加对话到记忆"""
        self._init_db()

        combined = f"用户: {user_msg}\n助手: {assistant_msg}"
        entry_id = hashlib.md5(combined.encode()).hexdigest()[:12]

        # 检查是否已存在
        if any(e.id == entry_id for e in self._entries):
            return

        entry = MemoryEntry(
            id=entry_id,
            content=combined,
            layer=MemoryLayer.SHORT_TERM.value,
            importance=self._calculate_importance(combined),
            timestamp=datetime.now().isoformat(),
            keywords=self.processor.tokenize(combined)[:20],
            metadata=metadata or {},
        )
        entry.vector = self.processor.transform(combined)

        self._entries.append(entry)
        self._save_to_db(entry)

        # 重建索引以包含新条目
        self._rebuild_index()

    def retrieve(self, query: str, top_k: int = 5,
                  layer: Optional[str] = None) -> List[MemoryEntry]:
        """语义检索相关记忆"""
        if not self._entries:
            return []

        query_vec = self.processor.transform(query)
        scored = []

        for entry in self._entries:
            if layer and entry.layer != layer:
                continue
            if not entry.vector:
                continue
            sim = TextProcessor.cosine_similarity(query_vec, entry.vector)
            # 结合重要性加权
            score = sim * 0.7 + entry.importance * 0.3
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:top_k] if s > 0.1]

    def search_keyword(self, keyword: str, top_k: int = 10) -> List[MemoryEntry]:
        """关键词精确搜索"""
        results = []
        for entry in self._entries:
            if keyword.lower() in entry.content.lower():
                results.append(entry)
        return results[:top_k]

    def consolidate(self, summarize_fn=None):
        """记忆巩固：将短期记忆压缩为长期记忆"""
        short_term = [e for e in self._entries if e.layer == MemoryLayer.SHORT_TERM.value]
        if len(short_term) < 3:
            return 0  # 不够，不巩固

        # 按时间分组（同一天）
        by_date = {}
        for entry in short_term:
            date = entry.timestamp[:10]
            by_date.setdefault(date, []).append(entry)

        consolidated = 0
        for date, entries in by_date.items():
            if len(entries) < 2:
                continue

            # 合并同类对话
            combined = "\n---\n".join(e.content for e in entries)
            summary = summarize_fn(combined) if summarize_fn else self._auto_summarize(combined)

            # 创建长期记忆条目
            long_entry = MemoryEntry(
                id=hashlib.md5(f"{date}_consolidated".encode()).hexdigest()[:12],
                content=combined,
                summary=summary,
                layer=MemoryLayer.LONG_TERM.value,
                importance=max(e.importance for e in entries),
                timestamp=date + "T23:59:59",
                keywords=self.processor.tokenize(combined)[:30],
                metadata={"consolidated_from": [e.id for e in entries], "date": date},
            )
            long_entry.vector = self.processor.transform(combined)

            # 删除原始短期记忆
            ids_to_delete = {e.id for e in entries}
            self._entries = [e for e in self._entries if e.id not in ids_to_delete]
            conn = self._get_conn()
            try:
                for eid in ids_to_delete:
                    conn.execute("DELETE FROM memories WHERE id = ?", (eid,))
                conn.commit()
            finally:
                if self.db_path != ":memory:":
                    conn.close()
            self._entries.append(long_entry)
            self._save_to_db(long_entry)
            consolidated += 1

        self._rebuild_index()
        return consolidated

    def _auto_summarize(self, content: str) -> str:
        """自动摘要（提取关键句）"""
        sentences = re.split(r'[。！？\n]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 2:
            return content[:200]
        # 取前两句 + 后一句
        return "。".join(sentences[:2] + sentences[-1:])

    def _calculate_importance(self, content: str) -> float:
        """计算记忆重要性"""
        score = 0.5  # 基础分

        # 长度加分（太短可能是闲聊）
        if len(content) > 100:
            score += 0.1
        if len(content) > 300:
            score += 0.1

        # 关键词密度
        tokens = self.processor.tokenize(content)
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
        score += unique_ratio * 0.2

        # 是否含问题（用户提问通常更重要）
        if '?' in content or '？' in content:
            score += 0.1

        return min(1.0, max(0.1, score))

    # ─── 记忆管理 ───

    def get_stats(self) -> Dict[str, Any]:
        """记忆统计"""
        layers = Counter(e.layer for e in self._entries)
        return {
            "total": len(self._entries),
            "by_layer": dict(layers),
            "avg_importance": sum(e.importance for e in self._entries) / len(self._entries) if self._entries else 0,
            "oldest": min(e.timestamp for e in self._entries) if self._entries else None,
            "newest": max(e.timestamp for e in self._entries) if self._entries else None,
        }

    def get_context_string(self, query: str, max_tokens: int = 2000) -> str:
        """获取注入到 LLM 的上下文字符串"""
        results = self.retrieve(query, top_k=5)
        if not results:
            return ""

        parts = ["【相关记忆】"]
        for entry in results:
            summary = entry.summary or entry.content[:200]
            parts.append(f"- [{entry.timestamp[:10]}] {summary}")

        context = "\n".join(parts)
        # 粗略截断
        if len(context) > max_tokens:
            context = context[:max_tokens]
        return context

    def clear_layer(self, layer: str):
        """清除指定层的记忆"""
        self._entries = [e for e in self._entries if e.layer != layer]
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM memories WHERE layer = ?", (layer,))
            conn.commit()
        finally:
            if self.db_path != ":memory:":
                conn.close()
        self._rebuild_index()

    def export_json(self, path: str):
        """导出为 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)

    def import_json(self, path: str):
        """从 JSON 导入"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data:
            entry = MemoryEntry.from_dict(d)
            if not any(e.id == entry.id for e in self._entries):
                self._entries.append(entry)
                self._save_to_db(entry)
        self._rebuild_index()


# ─── 演示 ───

def demo():
    """内置演示"""
    mem = AgentMemory()

    # 添加一些对话
    mem.add_conversation(
        "什么是机器学习？",
        "机器学习是人工智能的一个分支，让计算机从数据中学习模式，无需显式编程。"
    )
    mem.add_conversation(
        "深度学习和机器学习有什么区别？",
        "深度学习是机器学习的子集，使用神经网络（多层）来学习数据的层次化特征。"
    )
    mem.add_conversation(
        "今天天气怎么样？",
        "今天北京晴，气温 15-22 度，适合户外活动。"
    )
    mem.add_conversation(
        "推荐一本 AI 入门的书",
        "推荐《深度学习》（花书）by Goodfellow、Bengio、Courville，以及《机器学习》by 周志华。"
    )

    print("=== 记忆统计 ===")
    stats = mem.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== 语义检索: 'AI 入门学习' ===")
    results = mem.retrieve("AI 入门学习", top_k=3)
    for entry in results:
        print(f"  [{entry.layer}] [{entry.timestamp[:10]}] {entry.content[:80]}... (重要性: {entry.importance:.2f})")

    print("\n=== 关键词搜索: '深度' ===")
    results = mem.search_keyword("深度", top_k=3)
    for entry in results:
        print(f"  {entry.content[:80]}...")

    print("\n=== 上下文注入 ===")
    ctx = mem.get_context_string("如何学习人工智能")
    print(ctx[:300])

    print("\n=== 记忆巩固 ===")
    consolidated = mem.consolidate()
    print(f"  巩固了 {consolidated} 条长期记忆")

    print("\n✅ 演示完成")


if __name__ == "__main__":
    demo()