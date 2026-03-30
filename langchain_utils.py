from langchain_community.llms.tongyi import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.load import dumps, loads
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba
from typing import List
from db_utils import get_chat_history
from chroma_utils import vectorstore
from dotenv import load_dotenv

load_dotenv()

llm = Tongyi(model="qwen-turbo", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def Query_Rewrie(session_id: str, query: str):
    template = """
    You are a smart query rewriting assistant. 
    Rewrite the user's query to make it clear, concise, and optimized for information retrieval or downstream processing based on the given context. 
    Keep the meaning the same, remove ambiguity, and make it more specific if possible. 
    Do not add extra information that wasn't implied in the original query.

    Original Query: "{query}"
    context: {context}
    Rewritten Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    context = get_chat_history(session_id)

    rewrite_query = chain.invoke({"query": query, "context": context})

    return rewrite_query

def generate_multi_queries(query: str):
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {query}
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    chain = prompt_perspectives | llm | StrOutputParser()

    queries = chain.invoke({"query": query})

    return queries.split("\n")

# RAG-Fusion 核心算法： Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results: list[list], k=60):
    """ 融合多个查询的结果，通过排名加权计算文档分数，并得到最终的 reranked 文档列表 
        score(d) = sum(1 / (k + rank(d, q_i))) for i in range(len(queries))
    """

    # 初始化分数字典
    fused_scores = {}

    # 遍历每个查询的结果
    for docs in results:
        # 遍历每个文档及其排名
        for rank, doc in enumerate(docs):
            # 转换成字符串
            doc_str = dumps(doc)

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            previous_score = fused_scores[doc_str]
            # 通过 RRF 公式更新分数
            fused_scores[doc_str] += 1 / (k + rank)
    
    # 将文档字符串转换回对象，并按分数排序
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    ]

    return reranked_results

# 提取 query 中的关键词，进行基于关键词的 rerank
def rerank(query: str, docs: list) -> List[Document]:
    # 分词查询
    tokenized_query = list(jieba.cut(query))
    
    # 提取文档内容
    corpus = [doc.page_content for doc in docs]
    
    # 分词文档
    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    
    # 创建BM25模型
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 计算分数
    scores = bm25.get_scores(tokenized_query)
    
    # 排序文档
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    # 返回排序后的文档
    return [doc for doc, score in scored_docs]