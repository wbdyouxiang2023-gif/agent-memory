"""
Agent Memory System 测试

覆盖核心功能：添加、检索、巩固、统计、导入导出。
"""

import os
import sys
import tempfile
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_memory import AgentMemory, MemoryLayer, TextProcessor


def test_add_and_retrieve():
    """测试添加对话和语义检索"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("什么是机器学习？", "机器学习是AI的分支，让计算机从数据中学习。")
    mem.add_conversation("深度学习是什么？", "深度学习使用神经网络来学习层次化特征。")

    assert len(mem._entries) == 2

    results = mem.retrieve("机器学习", top_k=2)
    assert len(results) >= 1
    assert "机器学习" in results[0].content
    print("✅ test_add_and_retrieve")


def test_keyword_search():
    """测试关键词搜索"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("今天天气很好", "北京晴天，20度。")
    mem.add_conversation("明天会下雨吗", "预报说明天有雨。")

    results = mem.search_keyword("天气")
    assert len(results) == 1
    assert "天气" in results[0].content
    print("✅ test_keyword_search")


def test_importance_scoring():
    """测试重要性评分"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("hi", "hello")  # 短对话，低重要性
    mem.add_conversation(
        "请详细解释Transformer架构的自注意力机制和多头注意力的区别",
        "Transformer的自注意力机制通过Q、K、V三个矩阵计算序列中每个位置对其他位置的注意力权重。多头注意力将这个过程并行执行多次，每个头关注不同的表示子空间，最后拼接所有头的输出。这样模型可以同时关注位置、语义和句法等不同层面的信息。"
    )  # 长对话，高重要性

    entries = mem._entries
    assert entries[1].importance > entries[0].importance
    print(f"✅ test_importance_scoring (短: {entries[0].importance:.2f}, 长: {entries[1].importance:.2f})")


def test_consolidate():
    """测试记忆巩固"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    # 添加多条同一天的对话
    for i in range(5):
        mem.add_conversation(f"问题{i}", f"回答{i}，这是第{i}个关于机器学习的对话。")

    consolidated = mem.consolidate()
    assert consolidated >= 1

    # 检查长期记忆层有内容
    long_entries = [e for e in mem._entries if e.layer == MemoryLayer.LONG_TERM.value]
    assert len(long_entries) >= 1
    assert long_entries[0].summary  # 有摘要
    print(f"✅ test_consolidate (巩固了 {consolidated} 条)")


def test_stats():
    """测试统计信息"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("测试问题", "测试回答")
    stats = mem.get_stats()

    assert stats["total"] == 1
    assert "by_layer" in stats
    assert "avg_importance" in stats
    print(f"✅ test_stats ({stats})")


def test_context_string():
    """测试上下文注入"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("什么是RAG？", "RAG是检索增强生成，结合了信息检索和文本生成。")
    mem.add_conversation("今天天气怎么样？", "今天晴天，22度。")

    ctx = mem.get_context_string("RAG检索增强", max_tokens=500)
    assert "RAG" in ctx
    assert "检索增强" in ctx
    print(f"✅ test_context_string ({len(ctx)} chars)")


def test_export_import():
    """测试导入导出"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("测试导出", "导出成功")
    original_count = len(mem._entries)

    # 导出
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        export_path = f.name
    mem.export_json(export_path)

    # 导入到新实例
    mem2 = AgentMemory(db_path=":memory:")
    mem2._init_db()
    mem2.import_json(export_path)

    assert len(mem2._entries) == original_count
    os.unlink(export_path)
    print("✅ test_export_import")


def test_clear_layer():
    """测试清除记忆层"""
    mem = AgentMemory(db_path=":memory:")
    mem._init_db()

    mem.add_conversation("测试1", "回答1")
    # 手动创建一条长期记忆
    from agent_memory import MemoryEntry
    entry = MemoryEntry(
        id="test_long",
        content="长期记忆内容",
        layer="long",
        timestamp="2026-01-01T00:00:00",
    )
    mem._entries.append(entry)
    mem._save_to_db(entry)

    assert len(mem._entries) == 2
    mem.clear_layer("long")
    assert len(mem._entries) == 1
    print("✅ test_clear_layer")


def test_cosine_similarity():
    """测试余弦相似度"""
    vec1 = {"a": 0.5, "b": 0.5, "c": 0.0}
    vec2 = {"a": 0.5, "b": 0.5, "d": 0.0}
    sim = TextProcessor.cosine_similarity(vec1, vec2)
    assert sim > 0.9  # 高度相似

    vec3 = {"x": 1.0, "y": 0.0}
    sim2 = TextProcessor.cosine_similarity(vec1, vec3)
    assert sim2 == 0.0  # 完全不相似
    print("✅ test_cosine_similarity")


def test_text_processor():
    """测试文本处理器"""
    tp = TextProcessor()
    tp.fit(["机器学习是AI的分支", "深度学习使用神经网络"])

    vec = tp.transform("机器学习")
    assert "机器" in vec or "器学" in vec or "机器学习" in vec

    vec2 = tp.transform("深度学习")
    sim = TextProcessor.cosine_similarity(vec, vec2)
    assert sim >= 0.0  # 可以是任意值，但不能报错
    print("✅ test_text_processor")


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_add_and_retrieve,
        test_keyword_search,
        test_importance_scoring,
        test_consolidate,
        test_stats,
        test_context_string,
        test_export_import,
        test_clear_layer,
        test_cosine_similarity,
        test_text_processor,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"通过: {passed}/{len(tests)}")
    if failed > 0:
        print(f"失败: {failed}")
        sys.exit(1)
    else:
        print("🎉 全部通过！")


if __name__ == "__main__":
    run_all_tests()