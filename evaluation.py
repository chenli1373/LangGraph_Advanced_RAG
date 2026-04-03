import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from chroma_utils import embedding_function

# 基础的相似函数
def compute_similarity(text1: str, text2: str) -> float:
    emb = embedding_function.encode([text1, text2])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

    return float(sim)

# Rewrite 评分
def score_rewrite(query: str, rewritten_query: str) -> float:
    sim = compute_similarity(query, rewritten_query)

    # 惩罚 完全没改写
    if query.strip().lower() == rewritten_query.strip().lower():
        sim *= 0.8
    
    return sim

# MUlti-query 多样性评分
def score_multi_query(queries: List[str]) -> float:
    # 需要生成多个 query，如果 query 太少则认为多样性不足
    if len(queries) < 2:
        return 0.0
    
    embeddings = embedding_function.encode(queries)
    sim_matrix = cosine_similarity(embeddings)

    # 去掉对角线（自己和自己）
    mask = ~np.eye(len(queries), dtype=bool)
    avg_sim = sim_matrix[mask].mean()

    diversity = 1 - avg_sim  # 多样性越高，分数越高
    return float(diversity)

# Retrieval 评分
def score_retrieval(query: str, docs: List[str]) -> float:
    if not docs:
        return 0.0
    
    contents = [doc.page_content for doc in docs]

    q_emb = embedding_function.encode([query])
    d_emb = embedding_function.encode(contents)

    sim = cosine_similarity(q_emb, d_emb)[0]

    # 取最大值，至少有正确信息，如果用 mean() 可能会被大量无关文档拉低分数
    return float(np.max(sim))

# Rerank 评分
def score_rerank(query: str, docs: List[str]) -> float:
    if not docs:
        return 0.0
    
    contents = [doc.page_content for doc in docs[: 3]]

    q_emb = embedding_function.encode([query])
    d_emb = embedding_function.encode(contents)

    sims = cosine_similarity(q_emb, d_emb)[0]

    return float(np.mean(sims))

# Answer 评分
def score_answer(query: str, answer: str, docs: List[str]) -> float:
    if not answer:
        return 0.0
    
    context = " ".join([doc.page_content for doc in docs[: 3]])

    q_emb = embedding_function.encode([query])
    a_emb = embedding_function.encode([answer])
    c_emb = embedding_function.encode([context])

    # 计算答案与查询和上下文的相似度
    sim_answer_query = cosine_similarity(a_emb, q_emb)[0][0]
    sim_answer_context = cosine_similarity(a_emb, c_emb)[0][0]

    # 综合评分
    score = 0.5 * sim_answer_query + 0.5 * sim_answer_context
    return float(score)

# 规则评分（防止胡编）
def score_rules(answer: str, docs: List[str]) -> float:
    if not answer:
        return 0.0
    
    context = " ".join([doc.page_content for doc in docs[: 3]])

    score = 1.0

    # 是否使用过 context
    # 语义相似度
    a_emb = embedding_function.encode([answer])
    c_emb = embedding_function.encode([context])
    sim_answer_context = cosine_similarity(a_emb, c_emb)[0][0]

    if sim_answer_context < 0.3:
        score *= 0.5  # 基本没用 context，可能是胡编
    elif sim_answer_context < 0.6:
        score *= 0.8  # 有一定使用 context，但不够充分
    
    # 关键词覆盖
    context_keywords = set(context.split())
    answer_words = set(answer.split())

    overlap = sum(1 for word in answer_words if word in context_keywords)
    ratio = overlap / (len(answer_words) + 1e-6)

    if ratio < 0.2:
        score *= 0.5  # 关键词覆盖率太低，可能是胡编
    elif ratio < 0.5:
        score *= 0.8  # 关键词覆盖率不够，可能部分胡编
    
    # 事实一致性，判断 answer 是否包含 context 中的关键片段
    hit = False
    for sentence in context.split("."):
        if len(sentence) > 20 and sentence.strip() in answer:
            hit = True
            break

    if not hit:
        score *= 0.7  # 没有明显使用 context 中的事实，可能是胡编

    # 不确定性表达惩罚
    uncertain_words = ["不知道", "不清楚", "无法确定", "可能是", "大概是", "maybe", "probably", "i think", "might"]
    if any(word in answer.lower() for word in uncertain_words):
        score *= 0.7

    # 长度惩罚，过短可能不完整，过长可能胡编
    if len(answer.split()) < 5:
        score *= 0.6
    
    return float(score)

# 最终评分
def compute_final_score(state: dict) -> float:
    return (
        0.1 * state.get("score_rewrite", 0) +
        0.1 * state.get("score_multi_query", 0) +
        0.2 * state.get("score_retrieval", 0) +
        0.2 * state.get("score_rerank", 0) +
        0.3 * state.get("score_answer", 0) +
        0.1 * state.get("score_rules", 0)
    )

def evaluation_all(state: dict) -> dict:
    query = state.get("query", "")
    rewritten_query = state.get("rewritten_query", "")
    queries = state.get("queries", [])
    retrieval_docs = state.get("retrieval_docs", [])
    reranked_docs = state.get("reranked_docs", [])
    answer = state.get("answer", "")

    state["score_rewrite"] = score_rewrite(query, rewritten_query)
    state["score_multi_query"] = score_multi_query(queries)
    state["score_retrieval"] = score_retrieval(query, retrieval_docs)
    state["score_rerank"] = score_rerank(query, reranked_docs)
    state["score_answer"] = score_answer(query, answer, reranked_docs)
    state["score_rules"] = score_rules(answer, reranked_docs)

    state["final_score"] = compute_final_score(state)

    return state