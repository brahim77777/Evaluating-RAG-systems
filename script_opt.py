"""Hybrid RAG — Rust modules called from Python.

Original functionality preserved:
  - PDF hash caching (skip re-embed if PDFs unchanged)
  - Benchmark mode (BENCHMARK_ITERS x BENCHMARK_QUERIES)
  - LLM chat via Ollama (RUN_LLM flag)
  - log_run_info, all config flags

Optimizations added vs previous version:
  - embed_texts_rust() replaces SentenceTransformer — uses fastembed ONNX
    runtime in Rust (now with BGE Small EN v1.5)
  - lancedb_search no longer calls open_table on every run — Table handle
    is now cached in lib.rs via a global OnceCell, invalidated on each insert
  - RUN_PROFILING flag (default True) to toggle the 100-run loop
    CSV columns: run, pdf_read_ms, chunking_ms, model_embedding_ms,
                 db_insert_ms, search_ms
"""
import os
import sys
import io
import time
import csv
import hashlib
import json
from pathlib import Path
import torch
import requests
from dotenv import load_dotenv

from agents import Generator, Evaluator, QueryRefiner, Retriever, UserProxy

import rag_rust

# --- AJUSTEMENT POUR ÉVITER LA SUR-SOUSCRIPTION (i7-6700HQ) ---
# IMPORTANT: ces variables doivent être définies avant l'import de rag_rust.
os.environ.setdefault("EMBED_THREADS", "6")
os.environ.setdefault("EMBED_CHUNK_SIZE", "32")
os.environ.setdefault("EMBED_BATCH_SIZE", "32")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAX_THREADS", "1")


# Désactive les flags de télémétrie ONNX qui peuvent ralentir l'initialisation
os.environ["ONNXRUNTIME_FLAGS"] = "0" 
# ----------------------------------------


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {value!r}") from exc


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value!r}") from exc

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT          = 60
OLLAMA_CHAT_MODEL       = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("MODEL", "llama3.2")
EMBED_MODEL_NAME        = os.getenv("EMBED_MODEL_NAME") or "BAAI/bge-small-en-v1.5"
CHAT_TEMPERATURE        = get_env_float("CHAT_TEMPERATURE", 0.2)
TOP_K                   = get_env_int("TOP_K", 3)

# Storage settings
DB_DIR     = os.getenv("DB_DIR", "lancedb")
TABLE_NAME = "pdf_chunks"
REBUILD_DB = True

# Cache settings
CACHE_FILE = ".rag_cache.json"

# PDF ingestion settings
PDF_DIR       = os.getenv("PDF_FOLDER", "pdfs")
PDF_PATHS     = []
MAX_PAGES     = None
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# Runtime options
RUN_LLM         = True
RUN_BENCHMARK   = False
BENCHMARK_ITERS = 20
BENCHMARK_QUERIES = [
    "Summarize the main topic.",
    "What are the key conclusions?",
    "List important definitions.",
]

# ── Profiling options ──────────────────────────────────────────────────────────
RUN_PROFILING = False
NUM_RUNS      = 100
CSV_FILE      = "profile_results.csv"
CSV_HEADER    = ["run", "pdf_read_ms", "chunking_ms", "model_embedding_ms",
                 "db_insert_ms", "search_ms"]

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUESTION = (
    "Question: According to the abstract, what specific type of 'framework' "
    "does this paper propose to support knowledge management and decision-making?"
)

start_time = time.perf_counter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_pdf_hash() -> str:
    paths = []
    if PDF_PATHS:
        paths = [Path(p) for p in PDF_PATHS]
    else:
        pdf_dir = Path(PDF_DIR)
        if pdf_dir.exists():
            paths = sorted(pdf_dir.glob("*.pdf"))

    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBED_MODEL_NAME}".encode())
    return h.hexdigest()


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(pdf_hash: str, chunks: list) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump({"hash": pdf_hash, "chunks": chunks}, f)


def ollama_post(path, payload, retries=3):
    for attempt in range(retries):
        try:
            url = f"{OLLAMA_BASE_URL}{path}"
            response = requests.post(
                url, json=payload, timeout=OLLAMA_TIMEOUT
            )

            if not response.ok:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(f"Ollama error {response.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
                if attempt < retries - 1:
                    print(f"Connection error, retrying ({attempt+1}/{retries})...")
                    time.sleep(2)
                else:
                    raise
    return response.json()


# def embed_texts(texts: list) -> list:
#     """Embed via fastembed ONNX runtime in Rust — same engine as rust_only."""
#     return rag_rust.embed_texts_rust(texts)
# ── Dans script_opt.py ────────────────────────────────────────────────────────

def embed_texts(texts: list) -> list:
    """Ajoute le préfixe BGE et envoie le batch complet à Rust."""
    # BGE-small nécessite un préfixe pour les passages indexés
    prefixed_texts = [f"passage: {t}" for t in texts]
    return rag_rust.embed_texts_rust(prefixed_texts)

def warmup_embedder() -> None:
    """Warm up embedding threads to avoid counting model init time."""
    try:
        warmup_iters = int(os.getenv("EMBED_WARMUP", "1"))
    except ValueError:
        warmup_iters = 1
    if warmup_iters <= 0:
        return

    try:
        warmup_batch = int(os.getenv("EMBED_CHUNK_SIZE", "64"))
    except ValueError:
        warmup_batch = 64
    warmup_batch = max(1, warmup_batch)

    print(f"Warming up embedder ({warmup_iters}x{warmup_batch})...")
    for _ in range(warmup_iters):
        rag_rust.embed_texts_rust(["warmup"] * warmup_batch)

def build_or_open_table(chunks: list, needs_rebuild: bool) -> None:
    if not needs_rebuild:
        print("DB is up-to-date, skipping.")
        return

    print(f"Embedding {len(chunks)} chunks...")
    warmup_embedder()
    embed_start = time.perf_counter()
    
    # APPEL UNIQUE : On envoie la liste complète ici
    embeddings = embed_texts(chunks) 
    
    embed_time = time.perf_counter() - embed_start
    print(f"Embedding time: {embed_time*1000:.2f}ms ({len(chunks)/embed_time:.2f} chunks/s)")

    rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)

def chat_complete(messages):
    data = ollama_post(
        "/api/chat",
        {
            "model": OLLAMA_CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": CHAT_TEMPERATURE},
        },
    )
    content = (data.get("message") or {}).get("content", "")
    model_used = data.get("model") or OLLAMA_CHAT_MODEL
    return content, model_used


def load_pdf_texts() -> list:
    paths = []
    if PDF_PATHS:
        paths = [Path(p) for p in PDF_PATHS]
    else:
        pdf_dir = Path(PDF_DIR)
        if pdf_dir.exists():
            paths = list(pdf_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError("No PDFs found.")
    try:
        pages = rag_rust.load_pdf_pages_many([str(p) for p in paths])
    except Exception as exc:
        raise RuntimeError("Failed to load PDFs via Rust.") from exc
    if MAX_PAGES is not None:
        pages = pages[:MAX_PAGES]
    return [t for t in pages if t and t.strip()]


def chunk_texts(pages: list) -> list:
    chunks = []
    for page_text in pages:
        if page_text and page_text.strip():
            chunks.extend(
                c for c in rag_rust.smart_chunker(page_text, CHUNK_SIZE, CHUNK_OVERLAP) if c
            )
    return chunks


def log_run_info():
    print("=== Run Info ===")
    print(f"Chat model   : {OLLAMA_CHAT_MODEL}")
    print(f"Ollama URL   : {OLLAMA_BASE_URL}")
    print(f"Embed model  : {EMBED_MODEL_NAME}")
    print(f"Chunk size   : {CHUNK_SIZE} | overlap: {CHUNK_OVERLAP}")
    print(
        "Embed config: threads=%s chunk=%s batch=%s"
        % (
            os.getenv("EMBED_THREADS"),
            os.getenv("EMBED_CHUNK_SIZE"),
            os.getenv("EMBED_BATCH_SIZE"),
        )
    )
    print("Embed engine : fastembed ONNX (Rust)")
    print("================")


# def build_or_open_table(chunks: list, needs_rebuild: bool) -> None:
#     if not needs_rebuild:
#         print("DB is up-to-date, skipping embed + insert.")
#         return

#     embed_start = time.perf_counter()
#     embeddings = embed_texts(chunks)
#     embed_time = time.perf_counter() - embed_start
#     print(f"Embedding time: {embed_time*1000:.2f}ms ({len(chunks)/embed_time:.2f} chunks/s)")

#     rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)


def retrieve(query: str, top_k: int = TOP_K):
    prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
    query_embedding = embed_texts([prefixed_query])[0]
    return rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_embedding, top_k)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # torch.set_num_threads(os.cpu_count())

    # Collect PDF paths — used by both profiling loop and original pipeline
    if PDF_PATHS:
        all_pdf_paths = sorted(str(Path(p)) for p in PDF_PATHS)
    else:
        all_pdf_paths = sorted(str(p) for p in Path(PDF_DIR).glob("*.pdf"))

    if not all_pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    print(f"Found {len(all_pdf_paths)} PDF(s): {[Path(p).name for p in all_pdf_paths]}")

    # Pre-load fastembed model once (triggers OnceCell in Rust, mirrors rust_only)
    print("Loading fastembed model (once)...")

    rag_rust.load_embed_model()
    

    # ══════════════════════════════════════════════════════════════════════════
    # PROFILING LOOP — 100 runs, CSV format identical to rust_only / python_only
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_PROFILING:
        csv_path = Path(CSV_FILE)
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

        profiling_context = ""

        for run in range(1, NUM_RUNS + 1):
            print(f"\n{'─' * 50}")
            print(f"Run {run}/{NUM_RUNS}")

            # 1. PDF read & extract — Rust Rayon (par_iter over files + pages) ─
            t0 = time.perf_counter()
            pages = rag_rust.load_pdf_pages_many(all_pdf_paths)
            pages = [p for p in pages if p and p.strip()]
            pdf_read_ms = (time.perf_counter() - t0) * 1000
            print(f"  PDF Read & Extract:    {pdf_read_ms:.4f} ms  ({len(all_pdf_paths)} file(s), {len(pages)} pages)")

            # 2. Text chunking — Rust smart_chunker ───────────────────────────
            t0 = time.perf_counter()
            chunks = []
            for page_text in pages:
                chunks.extend(
                    c for c in rag_rust.smart_chunker(page_text, CHUNK_SIZE, CHUNK_OVERLAP) if c
                )
            chunking_ms = (time.perf_counter() - t0) * 1000
            print(f"  Text Chunking:         {chunking_ms:.4f} ms  ({len(chunks)} chunks)")

            # 3. Embedding — fastembed ONNX via Rust (same engine as rust_only) ─
            t0 = time.perf_counter()
            embeddings = embed_texts(chunks)
            embedding_ms = (time.perf_counter() - t0) * 1000
            print(f"  Model/Embedding:       {embedding_ms:.4f} ms")

            # 4. DB insertion — Rust LanceDB bindings (table cache invalidated) ─
            t0 = time.perf_counter()
            rag_rust.lancedb_create_or_open(DB_DIR, TABLE_NAME, chunks, embeddings, True)
            db_insert_ms = (time.perf_counter() - t0) * 1000
            print(f"  DB Init & Insertion:   {db_insert_ms:.4f} ms  ({len(chunks)} rows)")

            # 5. Vector search — Rust LanceDB (cached Table handle, no open_table)
            t0 = time.perf_counter()
            query_vec = embed_texts([f"{BGE_QUERY_PREFIX}{QUESTION}"])[0]
            
            results = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vec, TOP_K)
            profiling_context = "\n\n".join(text for text, _ in results)
            search_ms = (time.perf_counter() - t0) * 1000
            print(f"  Search:                {search_ms:.4f} ms")

            # Append to CSV
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    run,
                    f"{pdf_read_ms:.4f}",
                    f"{chunking_ms:.4f}",
                    f"{embedding_ms:.4f}",
                    f"{db_insert_ms:.4f}",
                    f"{search_ms:.4f}",
                ])

        # Single LLM call after profiling (not profiled, mirrors other impls)
        print(f"\n{'─' * 50}")
        print("Running LLM query after profiling (single call, not profiled)...")
        try:
            response_text, model_used = chat_complete([
                {"role": "user", "content": (
                    f"Use the following context to answer the question.\n\n"
                    f"Context:\n{profiling_context}\n\nQuestion: {QUESTION}"
                )}
            ])
            print(f"\nAnswer: {response_text}")
            if model_used:
                print(f"Model used: {model_used}")
        except Exception as exc:
            print(f"LLM call failed: {exc}")

        print(f"\nProfiling complete. Results saved to '{CSV_FILE}'.")

    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL PIPELINE (hash-cached, full features)
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Hash PDFs to detect changes
    current_hash = compute_pdf_hash()
    cache = load_cache()
    db_exists = Path(DB_DIR).exists()
    needs_rebuild = REBUILD_DB or not db_exists or cache.get("hash") != current_hash

    # 2. Load + chunk
    if needs_rebuild:
        pdf_start = time.perf_counter()
        pages = load_pdf_texts()
        print(f"Loaded {len(pages)} pages in {(time.perf_counter()-pdf_start)*1000:.2f}ms")

        chunk_start = time.perf_counter()
        dataset = chunk_texts(pages)
        print(f"Chunked into {len(dataset)} chunks in {(time.perf_counter()-chunk_start)*1000:.2f}ms")
        if dataset:
            avg_len = sum(len(c) for c in dataset) / len(dataset)
            print(f"Avg chunk length: {avg_len:.1f} chars")

        save_cache(current_hash, dataset)
    else:
        dataset = cache["chunks"]
        print(f"Cache hit: {len(dataset)} chunks loaded instantly, skipping PDF+embed.")

    # 3. Embed + insert only when needed
    build_start = time.perf_counter()
    build_or_open_table(dataset, needs_rebuild)
    print(f"DB build/open time: {(time.perf_counter()-build_start)*1000:.2f}ms")

    log_run_info()

    # 4. Query
    # input_query = "what is Samyama?"
    input_query = input("Ask your question...: ")

    if not input_query:
        input_query = "What is this document about?"

    state = {
        "query" : input_query,
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
    state = proxy.run(state)

    print(f"\nChatbot response:\n{state['answer']}")
    print(f"Score: {state['score']} | Attempts: {state['attempts']}")
    print(f"score: {state['score']}| retry: {state['should_retry']}")
    
    # 5. Benchmark
    if RUN_BENCHMARK:
        total_queries = BENCHMARK_ITERS * len(BENCHMARK_QUERIES)
        bench_start = time.perf_counter()
        for _ in range(BENCHMARK_ITERS):
            for q in BENCHMARK_QUERIES:
                retrieve(q)
        bench_total = time.perf_counter() - bench_start
        print(f"Benchmark: {bench_total*1000:.2f}ms for {total_queries} queries")
        print(f"Avg per query: {bench_total/total_queries:.4f}s")
