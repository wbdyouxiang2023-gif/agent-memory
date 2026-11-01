# Agent Memory System

轻量级 AI Agent 语义记忆引擎 — 让 Agent 真正拥有"记忆"。

## 是什么

一个纯 Python 标准库的 AI Agent 记忆系统。对话自动分块、语义检索、重要性评分、记忆巩固——无需外部 API，单文件部署。

```
用户输入 → Agent Memory → 记忆分层 → 语义检索 → 注入上下文
                     ↓
        工作记忆 / 短期记忆 / 长期记忆（自动巩固）
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔤 语义检索 | TF-IDF + 余弦相似度，理解"上次说的那个问题" |
| 📊 重要性评分 | 自动判断哪些对话值得记住 |
| 🏗️ 三层记忆 | 工作记忆 → 短期 → 长期（自动巩固压缩） |
| 💾 SQLite 持久化 | 跨会话持久化，支持 `:memory:` 模式 |
| 🔍 关键词+语义双搜索 | 精确匹配 + 语义关联 |
| 📤 导入导出 | JSON 格式，方便迁移 |

## 安装

```bash
git clone https://github.com/wbdyouxiang2023-gif/agent-memory.git
cd agent-memory
pip install -r requirements.txt  # 纯标准库，无需安装
```

## 快速使用

```python
from agent_memory import AgentMemory

mem = AgentMemory()

# 1. 添加对话
mem.add_conversation("什么是机器学习？", "机器学习是AI的分支...")
mem.add_conversation("深度学习呢？", "深度学习使用神经网络...")

# 2. 语义检索
results = mem.retrieve("AI 入门学习", top_k=3)
for entry in results:
    print(entry.content[:80])

# 3. 获取上下文（注入 LLM）
context = mem.get_context_string("如何学习人工智能")
print(context)

# 4. 记忆巩固（压缩短期→长期）
mem.consolidate()

# 5. 查看统计
stats = mem.get_stats()
print(stats)
```

## 记忆分层

```
┌─────────────────────────────────────┐
│  工作记忆 (Working)                  │  ← 当前会话
│  短期记忆 (Short-term, 7天)          │  ← 原始对话
│  长期记忆 (Long-term, 已巩固)        │  ← 摘要化压缩
└─────────────────────────────────────┘
         ↓ consolidate() 自动触发
   相关对话合并 → 摘要 → 升级为长期记忆
```

## 测试

```bash
python tests/test_memory.py
# 10/10 通过
```

## 路线图

- [ ] 集成真实嵌入模型（sentence-transformers）
- [ ] 支持多模态记忆（图片、语音）
- [ ] 记忆遗忘曲线（自动衰减不重要记忆）
- [ ] MCP 协议集成（让任何 Agent 都能用）

## License

MIT