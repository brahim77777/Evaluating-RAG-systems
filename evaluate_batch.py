    # evaluate_batch.py
from datasets import load_dataset
import time

from agents import UserProxy, QueryRefiner, Retriever, Generator, Evaluator
from script_opt import chat_complete, retrieve, TOP_K

def judge_answer(question, ground_truth, rag_answer):
    prompt = f"""
    You are an expert academic evaluator. 
    Question: {question}
    Ground Truth Answer: {ground_truth}
    RAG System Answer: {rag_answer}
    
    Did the RAG system correctly identify the core facts from the Ground Truth? 
    Score the RAG answer from 0 to 10 based on accuracy. Return ONLY an integer number.
    """
    messages = [{"role": "user", "content": prompt}]
    response, _ = chat_complete(messages)
    try:
        score = int(''.join(filter(str.isdigit, response)))
        return score / 10.0 
    except:
        return 0.0

def judge_retrieval(question, ground_truth, retrieved_chunks):
    safe_chunks = [str(chunk) for chunk in retrieved_chunks]
    combined_chunks = "\n---\n".join(safe_chunks)
    prompt = f"""
    You are an expert academic evaluator. 
    Question: {question}
    Ground Truth Answer: {ground_truth}
    
    Retrieved Context:
    {combined_chunks}
    
    Does the Retrieved Context contain the necessary facts to deduce the Ground Truth Answer? 
    Score from 0 to 10. Return ONLY an integer number.
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

NUM_PAPERS = 5
total_generation_score = 0
total_retrieval_score = 0
total_questions = 0

print(f"\n🚀 STARTING BATCH EVALUATION ON {NUM_PAPERS} PAPERS...\n")

for paper_idx in range(NUM_PAPERS):
    item = dataset[paper_idx]
    title = item['title']
    questions = item['qas']['question']
    answers_data = item['qas']['answers']
    
    print(f"==================================================")
    print(f"📄 PAPER {paper_idx + 1}: {title[:60]}...")
    print(f"==================================================")
    
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
                
        print(f"\n📝 Q: {query}")
        
        state = {
            "query": query, "answer": "", "score": 0.0, "attempts": 0,
            "chunks": [], "should_retry": False, "model_used": "", "refined_query": "" 
        }

        proxy = UserProxy(
            refiner=QueryRefiner(chat_fn=chat_complete),
            retriever=Retriever(retrieve_fn=retrieve, top_k=TOP_K),
            generator=Generator(chat_fn=chat_complete),
            evaluator=Evaluator(chat_fn=chat_complete, min_score=0.75), 
        )
        
        proxy.run(state)
        
        # Scoring
        retrieval_score = judge_retrieval(query, ground_truth, state['chunks'])
        gen_score = judge_answer(query, ground_truth, state['answer'])
        
        total_retrieval_score += retrieval_score
        total_generation_score += gen_score
        total_questions += 1
        
        print(f"🔎 Retrieval: {retrieval_score} | ⚖️ Generation: {gen_score} | 🔄 Attempts: {state['attempts']}")

print("\n" + "="*50)
print("🏆 GRAND AVERAGES ACROSS ALL PAPERS")
print("="*50)
print(f"Total Questions Evaluated: {total_questions}")
print(f"Average Context Recall (LanceDB): {total_retrieval_score / total_questions:.2f} / 1.0")
print(f"Average Answer Correctness (Agents): {total_generation_score / total_questions:.2f} / 1.0")