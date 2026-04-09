# evaluate_multi_agent.py
from datasets import load_dataset
import time

# Import your multi-agent classes and functions
from agents import UserProxy, QueryRefiner, Retriever, Generator, Evaluator
from script_opt import chat_complete, retrieve, TOP_K

def judge_answer(question, ground_truth, rag_answer):
    prompt = f"""
    You are an expert academic evaluator. 
    Question: {question}
    Ground Truth Answer: {ground_truth}
    RAG System Answer: {rag_answer}
    
    Did the RAG system correctly identify the core facts from the Ground Truth? 
    If the Ground Truth says "BIBREF19", and the RAG says "The text doesn't mention it", the score is 0.
    
    Score the RAG answer from 0 to 10 based on accuracy.
    Return ONLY an integer number between 0 and 10. No other text.
    """
    messages = [{"role": "user", "content": prompt}]
    response, _ = chat_complete(messages)
    try:
        score = int(''.join(filter(str.isdigit, response)))
        return score / 10.0 
    except:
        return 0.0

def judge_retrieval(question, ground_truth, retrieved_chunks):
    """Evaluates Context Recall: Did LanceDB fetch the right evidence?"""
    # FIX: Convert the LanceDB tuples into plain strings so we can join them safely
    safe_chunks = [str(chunk) for chunk in retrieved_chunks]
    combined_chunks = "\n---\n".join(safe_chunks)
    
    prompt = f"""
    You are an expert academic evaluator. 
    Question: {question}
    Ground Truth Answer: {ground_truth}
    
    Retrieved Context from Database:
    {combined_chunks}
    
    Does the Retrieved Context contain the necessary facts to deduce the Ground Truth Answer? 
    Score from 0 to 10. If the context is completely irrelevant or missing the facts, score 0.
    Return ONLY an integer number between 0 and 10. No other text.
    """
    messages = [{"role": "user", "content": prompt}]
    response, _ = chat_complete(messages)
    try:
        score = int(''.join(filter(str.isdigit, response)))
        return score / 10.0 
    except:
        return 0.0

print("Loading QASper dataset...")
dataset = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)
first_item = dataset[0] 

questions = first_item['qas']['question']
answers_data = first_item['qas']['answers']

print(f"\n--- Evaluating MULTI-AGENT System on: {first_item['title']} ---")
total_score = 0
retrieval_total_score = 0
generation_total_score = 0

for i in range(len(questions)):
    query = questions[i]
    
    ground_truth = "Unknown"
    if answers_data[i] and answers_data[i]['answer']:
        ans_dict = answers_data[i]['answer'][0]
        if ans_dict.get('free_form_answer'):
            ground_truth = ans_dict['free_form_answer']
        elif ans_dict.get('extractive_spans'):
            ground_truth = ", ".join(ans_dict['extractive_spans'])
        elif ans_dict.get('yes_no') is not None:
            ground_truth = str(ans_dict['yes_no'])
            
    print(f"\n📝 Question {i+1}: {query}")
    print(f"✅ Ground Truth: {ground_truth}")
    
    state = {
        "query" : query,
        "answer" : "",
        "score" : 0.0,
        "attempts" : 0,
        "chunks" : [],
        "should_retry" : False,
        "model_used" : "",
        "refined_query" : "" 
    }

    proxy = UserProxy(
        refiner=QueryRefiner(chat_fn=chat_complete),
        retriever=Retriever(retrieve_fn=retrieve, top_k=TOP_K),
        generator=Generator(chat_fn=chat_complete),
        evaluator=Evaluator(chat_fn=chat_complete, min_score=0.75), 
    )
    
    print("🤖 Agents are working (Refining -> Retrieving -> Generating -> Evaluating)...")
    start_time = time.time()
    state = proxy.run(state)
    end_time = time.time()
    
    print(f"🤖 RAG Answer ({end_time - start_time:.2f}s, Attempts: {state['attempts']}): {state['answer']}")
    print(f"🔄 Refined Query was: {state['refined_query']}")
    
    # Judge the Retrieval (LanceDB)
    retrieval_score = judge_retrieval(query, ground_truth, state['chunks'])
    retrieval_total_score += retrieval_score
    print(f"🔎 Context Recall (LanceDB Score): {retrieval_score} / 1.0")
    
    # Judge the final Generation (LLM Score)
    score = judge_answer(query, ground_truth, state['answer'])
    generation_total_score += score
    total_score += score
    print(f"⚖️  Answer Correctness Score: {score} / 1.0")
    print("-" * 50)

print(f"\n🏆 Final Multi-Agent Average Score: {total_score / len(questions):.2f} / 1.0")
print(f"\n🏆 Final Multi-Agent Average Generation Score: {generation_total_score / len(questions):.2f} / 1.0")
print(f"\n🏆 Final Multi-Agent Average Retrieval Score: {retrieval_total_score / len(questions):.2f} / 1.0")
