import re
from typing import Any, Dict, List, Optional

class Agent:
    def __init__(self, name: str):
        self.name = name
    
    def run(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name}.run() not implemented") 

    @staticmethod
    def _trace(state: Dict[str, Any], agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        trace = state.get("trace")
        if trace is None:
            trace = None
        item = {"agent": agent, "message": message}
        if data:
            item["data"] = data
        if trace is not None:
            trace.append(item)
        emit = state.get("emit")
        if emit:
            emit(item)

class UserProxy(Agent):
    def __init__(self, refiner, retriever, generator, evaluator):
        super().__init__("UserProxy")
        self.refiner = refiner
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator

    def run(self, state: dict) -> dict:
        # run once
        Agent._trace(state, self.name, "Starting agentic run.")
        state = self.refiner.run(state)
        state = self.retriever.run(state)
        state = self.generator.run(state)
        state = self.evaluator.run(state)

        # then retry if needed
        while state["should_retry"]:
            Agent._trace(
                state,
                self.name,
                "Retrying agentic run due to evaluator score.",
                {"attempts": state.get("attempts"), "score": state.get("score")},
            )
            state = self.refiner.run(state)
            state = self.retriever.run(state)
            state = self.generator.run(state)
            state = self.evaluator.run(state)

        return state

class Retriever(Agent):
    def __init__(self, retrieve_fn, top_k):
        super().__init__("Retriever")
        self.retrieve_fn = retrieve_fn
        self.top_k = top_k

    def run(self, state: dict) -> dict:
        query = state.get("refined_query") or state["query"]
        state["chunks"] = self.retrieve_fn(query, top_k=self.top_k)
        Agent._trace(
            state,
            self.name,
            "Retrieved chunks.",
            {"query_used": query, "top_k": self.top_k, "count": len(state["chunks"])},
        )
        return state




class Generator(Agent):
    def __init__(self, chat_fn):
        super().__init__("Generator")
        self.chat_fn = chat_fn
        
    def run(self, state: dict) -> dict:
        context = "\n".join(f" - {text}" for text, _ in state["chunks"])
        instruction_prompt = (
            "You are a helpful chatbot.\n"
            "Use only the following pieces of context to answer the question. "
            "Don't make up any new information:\n"
            f"{context}\n"
        )
        query = state.get("refined_query") or state["query"]
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        answer, model_used = self.chat_fn(messages)
        state["answer"] = answer
        state["model_used"] = model_used
        models = state.setdefault("models", {})
        models["generator"] = model_used
        Agent._trace(
            state,
            self.name,
            "Generated answer from retrieved context.",
            {"model_used": model_used, "context_chunks": len(state["chunks"])},
        )
        return state
    

class Evaluator(Agent):
    def __init__(self, chat_fn, min_score: float = 0.7, max_attempts: int = 3):
        super().__init__("Evaluator")
        self.chat_fn = chat_fn
        # On augmente le seuil d'exigence à 0.7 (70% de qualité minimum)
        self.min_score = min_score
        self.max_attempts = max_attempts
    
    def _score(self, query: str, answer: str, chunks: list) -> tuple[float, str, Optional[str]]:
        if not answer.strip():
            return 0.0, "Empty answer.", None
            
        context = "\n".join(f" - {text}" for text, _ in chunks)
        
        # Prompt "LLM-as-a-Judge" (short, shareable summary only)
        eval_prompt = (
            "You are a strict grading agent for a RAG system.\n"
            "Evaluate the quality of the 'Generated Answer' based on the 'User Query' and the 'Retrieved Context'.\n\n"
            "Criteria:\n"
            "- Faithfulness: Is the answer derived strictly from the context without hallucinations?\n"
            "- Relevance: Does the answer directly address the user's query?\n\n"
            f"User Query: {query}\n"
            f"Retrieved Context:\n{context}\n"
            f"Generated Answer:\n{answer}\n\n"
            "INSTRUCTIONS:\n"
            "1. First, write 'SUMMARY: ' followed by a single short sentence.\n"
            "2. Then, on a new line, write 'SCORE: ' followed by a float between 0.0 and 1.0.\n"
        )
        
        messages = [{"role": "user", "content": eval_prompt}]
        
        try:
            llm_response, model_used = self.chat_fn(messages)
            lines = llm_response.strip().split("\n")
            summary = "No summary provided"
            for line in lines:
                if line.strip().upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
                    break
            # Utilisation d'une regex pour extraire le score flottant de manière robuste
            # au cas où le LLM serait trop bavard (ex: "The score is 0.85")
            # Cherche "SCORE: 0.85" par exemple
            match = re.search(r"SCORE:\s*(0\.\d+|1\.0)", llm_response)
            if match:
                score = float(match.group(1))
            else:
                # Fallback de parsing direct
                score = 0.5 # neutral
                
            # Clamper la valeur entre 0.0 et 1.0 par sécurité
            return max(0.0, min(1.0, score)), summary, model_used
            
        except Exception as e:
            return 0.0, f"Evaluation failed: {e}", None
    
    def run(self, state: dict) -> dict:
        state["attempts"] += 1
        score, summary, model_used = self._score(state["query"], state["answer"], state["chunks"])
        state["score"] = score
        state["judge_summary"] = summary
        models = state.setdefault("models", {})
        models["evaluator"] = model_used
        
        # La condition de retry : score insuffisant ET on a encore des essais disponibles
        state["should_retry"] = (state["score"] < self.min_score) and (state["attempts"] < self.max_attempts)
        Agent._trace(
            state,
            self.name,
            "Evaluated answer quality.",
            {
                "score": state["score"],
                "summary": summary,
                "model_used": model_used,
                "should_retry": state["should_retry"],
                "attempts": state["attempts"],
            },
        )
        return state
    
class QueryRefiner(Agent):
    def __init__(self, chat_fn):
        super().__init__("QueryRefiner")
        self.chat_fn = chat_fn

    def run(self, state: dict) -> dict:
        prompt = (
            "Rewrite the user query to be clearer and more specific for retrieval. "
            "Keep the original intent, do not answer the query, and return only the rewritten query.\n\n"
            f"User query: {state['query']}"
        )
        messages = [{"role": "user", "content": prompt}]
        refined_query, model_used = self.chat_fn(messages)
        state["refined_query"] = refined_query.strip()
        models = state.setdefault("models", {})
        models["refiner"] = model_used
        Agent._trace(
            state,
            self.name,
            "Refined user query for retrieval.",
            {
                "original_query": state["query"],
                "refined_query": state["refined_query"],
                "model_used": model_used,
            },
        )
        return state
