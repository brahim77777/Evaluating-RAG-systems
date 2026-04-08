# evaluate.py
from datasets import load_dataset
from naive_rag import run_naive_pipeline
from script_opt import chat_complete # Import your LLM to act as the judge
import time

def judge_answer(question, ground_truth, rag_answer):
    """Uses the LLM to score the RAG answer from 0 to 10."""
    prompt = f"""
    You are an expert academic evaluator. 
    Question: {question}
    Ground Truth Answer: {ground_truth}
    RAG System Answer: {rag_answer}
    
    Did the RAG system correctly identify the core facts from the Ground Truth? 
    It is okay if the RAG system is more verbose, as long as it contains the facts.
    If the Ground Truth says "BIBREF19", and the RAG says "The text doesn't mention it", the score is 0.
    
    Score the RAG answer from 0 to 10 based on accuracy.
    Return ONLY an integer number between 0 and 10. No other text.
    """
    messages = [{"role": "user", "content": prompt}]
    response, _ = chat_complete(messages)
    
    # Clean up the response to just get the number
    try:
        score = int(''.join(filter(str.isdigit, response)))
        return score / 10.0 # Convert to a 0.0 - 1.0 scale
    except:
        return 0.0

print("Loading QASper dataset...")
dataset = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)
first_item = dataset[0] 

questions = first_item['qas']['question']
answers_data = first_item['qas']['answers']

print(f"\n--- Evaluating Paper: {first_item['title']} ---")
total_score = 0

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
    
    start_time = time.time()
    state = run_naive_pipeline(query)
    end_time = time.time()
    
    print(f"🤖 RAG Answer ({end_time - start_time:.2f}s): {state['answer']}")
    
    # --- NEW: Judge the answer ---
    score = judge_answer(query, ground_truth, state['answer'])
    total_score += score
    print(f"⚖️  LLM Judge Score: {score} / 1.0")
    print("-" * 50)

print(f"\n🏆 Final Average Score: {total_score / len(questions):.2f} / 1.0")