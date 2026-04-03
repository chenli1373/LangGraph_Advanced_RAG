import json
import os
from datetime import datetime

BAD_CASES_FILE = "AgenticRAG/bad_cases.json"

# 判断是否 bad case
def is_bad_case(state: dict) -> bool:
    return (
        state.get("final_score", 1.0) < 0.6   # 整体差
        or state.get("score_answer", 1.0) < 0.5  # 答案质量差
        or state.get("score_rules", 1.0) < 0.5  # 明显胡编
    )

# 保存 bad case
def save_bad_case(state: dict):
    data = {
        "time": str(datetime.now()),
        "query": state["query"],
        "rewritten_query": state["rewritten_query"],
        "queries": state["queries"],
        "retrieval_docs": [d.page_content for d in state.get("retrieval_docs", [])],
        "reranked_docs": [d.page_content for d in state.get("reranked_docs", [])],
        "answer": state["answer"],
        "scores": {
            "final": state["final_score"],
            "answer": state["score_answer"],
            "rules": state["score_rules"],
        }
    }

    # 文件不存在就创建
    if not os.path.exists(BAD_CASES_FILE):
        with open(BAD_CASES_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # 追加保存
    with open(BAD_CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    cases.append(data)

    # 写回
    with open(BAD_CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

# 构建 few-shot，用于 prompt
def build_few_shot_examples(max_examples=3):
    if not os.path.exists(BAD_CASES_FILE):
        return "No bad cases yet."
    
    with open(BAD_CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    # 取最新的 max_examples 个 bad case
    recent_cases = cases[-max_examples:]

    examples = []

    for case in recent_cases:
        example = f"""
        Query: {case['query']}
        Bad Answer: {case['answer']}
        Problems: Answer is low quality or not grounded in retrieved documents.
        """

        examples.append(example)

        return "\n".join(examples)