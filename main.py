from langgraph.graph import StateGraph, END
import graphviz
from typing import TypedDict
from langchain_utils import *
from db_utils import *
from chroma_utils import *
import os
import uuid

class RAGState(TypedDict):
    query: str
    session_id: str
    rewritten_query: str
    queries: str
    history: str
    retrieval_docs: str
    RRF_docs: str
    reranked_docs: str
    answer: str

# 保存已知用户 session_id 的字典
user_session = {} # {user_id: session_id}

# 准备工作，文档切分和索引，接收用户输入
def prepare_agent(state):
    allowed_extensions = [".pdf", ".docx", ".html"]
    file_path = input("Enter the path of the document to upload (PDF, DOCX, HTML): ").strip()
    while file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if file_extension not in allowed_extensions:
            print(f"Unsupported file type. Please upload a PDF, DOCX, or HTML file.")
            file_path = input("Enter the path of the document to upload (PDF, DOCX, HTML): ").strip()
            continue
        
        try:
            file_id = insert_document_record(file_name)
            success = index_document_to_chroma(file_path, file_id)

            if success:
                print(f"File {file_name} has been successfully uploaded and indexed, file_id: {file_id}")
            else:
                delete_document_record(file_id)
                raise Exception("Failed to index the document.")
        except Exception as e:
            print(f"Error processing the document: {e}")
            raise e

def user_input_agent(state):
    while True:
        user_id = input("Enter your user ID: ").strip()
        if not user_id:
            print("User ID cannot be empty. Please try again.")
            continue

        if user_id not in user_session:
            session_id = str(uuid.uuid4())
            user_session[user_id] = session_id
            print(f"New session created for user {user_id}, session_id: {session_id}")
        else:
            session_id = user_session[user_id]
            print(f"Existing session found for user {user_id}, session_id: {session_id}")
        break
    
    query = input("Enter your question or 'exit' to quit: ").strip()
    while query != "exit":
        if not query:
            print("No query entered. Please try again.")
            query = input("Enter your question or 'exit' to quit: ").strip()
        else: break
    if query == "exit":
        print("Exiting the Agentic RAG System. Goodbye!")
        exit(0)

    return {"query": query}

def query_rewrite_agent(state):
    query = state["query"]

    rewritten_query = Query_Rewrie(query)

    return {"rewritten_query": rewritten_query}

def multi_query_agent(state):
    query = state["query"]
    queries = generate_multi_queries(query)

    return {"queries": queries}

def retrieval_agent(state):
    queries = state["queries"]
    results = retriever.get_relevant_documents(queries)

    return {"retrieval_docs": results}

def fusion_agent(state):
    retrieval_docs = state["retrieval_docs"]
    fused_docs = reciprocal_rank_fusion(retrieval_docs)

    return {"RRF_docs": fused_docs}

def rerank_agent(state):
    query = state["query"]
    RRF_docs = state["RRF_docs"]
    reranked_docs = rerank(query, RRF_docs)

    return {"reranked_docs": reranked_docs}

def answer_agent(state):
    query = state["query"]
    reranked_docs = state["reranked_docs"]
    
    context = "\n".join([doc.page_content for doc in reranked_docs[:3]])

    prompt = f"""You are a helpful and precise assistant for answering questions based on the following retrieved documents.
    Use only the information from the retrieved documents to answer the question. If the retrieved documents do not
    contain enough information to answer the question, say you don't know. Do not try to fabricate an answer.
    Question: {query}
    Retrieved Documents:{context}
    """

    answer = llm(prompt).content
    insert_application_logs(state["session_id"], state["query"], answer, llm.model_name)
    print("=== AI Answer ===")
    print(answer)
    print("================\n")
    return {"answer": answer}

def continue_check_agent(state):
    cont = input("Do you want to ask another question? (yes/no): ").strip().lower()
    if cont == "yes":
        return "continue"
    else:
        print("Thank you for using the Agentic RAG System. Goodbye!")
        return "exit"

graph = StateGraph(RAGState)

graph.add_node("prepare", prepare_agent)
graph.add_node("user_input", user_input_agent)
graph.add_node("query_rewrite", query_rewrite_agent)
graph.add_node("multi_query", multi_query_agent)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("fusion", fusion_agent)
graph.add_node("rerank", rerank_agent)
graph.add_node("answer", answer_agent)
graph.add_node("router", lambda state: state)

graph.set_entry_point("prepare")

graph.add_edge("prepare", "user_input")
graph.add_edge("user_input", "query_rewrite")
graph.add_edge("query_rewrite", "multi_query")
graph.add_edge("multi_query", "retrieval")
graph.add_edge("retrieval", "fusion")
graph.add_edge("fusion", "rerank")
graph.add_edge("rerank", "answer")
graph.add_edge("answer", "router")
graph.add_conditional_edges(
    "router",
    continue_check_agent,
    {
        "continue": "prepare",
        "exit": END
    }
)

app = graph.compile()

# 可视化流程图
# png_data = app.get_graph().draw_png()
# with open("AgenticRAG/rag_graph.png", "wb") as f:
#     f.write(png_data)

app.invoke({})