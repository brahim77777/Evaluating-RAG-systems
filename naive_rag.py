# naive_rag.py
from agents import Retriever, Generator
# We import your existing functions from script_opt so we don't rewrite code
from script_opt import retrieve, chat_complete, TOP_K, log_run_info 

def run_naive_pipeline(query: str):
    # Setup the state exactly as your agents expect it, 
    # but we map 'refined_query' directly to the original query.
    state = {
        "query": query,
        "refined_query": query, 
        "answer": "",
        "score": 0.0,
        "attempts": 0,
        "chunks": [],
        "should_retry": False,
        "model_used": "",
    }

    # 1. Retrieve (using your Rust backend via the retrieve function)
    retriever = Retriever(retrieve_fn=retrieve, top_k=TOP_K)
    state = retriever.run(state)

    # 2. Generate (using your LLM)
    generator = Generator(chat_fn=chat_complete)
    state = generator.run(state)

    return state

# Quick test block
if __name__ == "__main__":
    test_query = input("Question: ")
    print(f"Running Naive RAG for: {test_query}")
    final_state = run_naive_pipeline(test_query)
    print("\n--- Answer ---")
    print(final_state["answer"])
