"""API server for the Rust/Python RAG stack.

Run:
  uvicorn api:app --reload
"""
from __future__ import annotations

import hashlib
import json
import os
os.environ["OMP_NUM_THREADS"] = "4"        
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAX_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Désactive les flags de télémétrie ONNX qui peuvent ralentir l'initialisation
os.environ["ONNXRUNTIME_FLAGS"] = "0" 
# ----------------------------------------
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import rag_rust
from agents import Evaluator, Generator, QueryRefiner, Retriever, UserProxy

APP_NAME = "Agentic-RAG-Rust-Core-PFE-26"

# Storage / DB
PDF_DIR = Path("data/pdfs")
META_PATH = Path("data/metadata.json")
DB_DIR = "lancedb"
TABLE_NAME = "pdf_chunks"

# Embeddings (Rust fastembed)
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT = 60
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("MODEL", "llama3.2")
CHAT_TEMPERATURE = 0.2

app = FastAPI(title=APP_NAME)

_INDEX_READY = False
_INDEX_STATUS = "idle"  # idle | building | ready | stale | error
_INDEX_INFO: Dict[str, Any] = {
  "last_build_at": None,
  "last_build_ms": None,
  "pages": None,
  "chunks": None,
  "last_error": None,
}
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
_EMBED_MODEL_READY = False

# CORS (frontend integration)
# _cors_allow_all = os.getenv("CORS_ALLOW_ALL", "false").lower() in {"1", "true", "yes"}
_cors_allow_all = True
_cors_origins = os.getenv("CORS_ORIGINS", "")
if _cors_allow_all:
  allow_origins = ["*"]
else:
  allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] or [
    "http://localhost:3000",
    "https://agentic-rag-rust-core-frontend-pfe.vercel.app/",
    "https://agentic-rag-rust-core-frontend-pfe-26-27avjnf8b.vercel.app",
    "https://agentic-rag-rust-core-frontend-pfe-26.vercel.app",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ]

app.add_middleware(
  CORSMiddleware,
  allow_origins=allow_origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class IndexRequest(BaseModel):
  rebuild: bool = True
  max_pages: Optional[int] = None


class QueryRequest(BaseModel):
  question: str
  top_k: int = 3
  chat_model: Optional[str] = None
  use_llm: bool = True
  mode: str = "agentic"  # "agentic" or "naive"
  return_trace: bool = True
  min_score: float = 0.7
  max_attempts: int = 3


class QueryResponse(BaseModel):
  answer: Optional[str]
  model_used: Optional[str]
  retrieved: List[dict]
  mode: Optional[str] = None
  refined_query: Optional[str] = None
  score: Optional[float] = None
  attempts: Optional[int] = None
  trace: Optional[List[dict]] = None
  models: Optional[Dict[str, Optional[str]]] = None


class DocumentMeta(BaseModel):
  filename: str
  size_bytes: int
  uploaded_at: Optional[str] = None
  updated_at: Optional[str] = None
  sha256: Optional[str] = None
  pages: Optional[int] = None


def ensure_dirs() -> None:
  PDF_DIR.mkdir(parents=True, exist_ok=True)
  META_PATH.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def load_metadata() -> Dict[str, Dict[str, Any]]:
  if not META_PATH.exists():
    return {}
  try:
    raw = META_PATH.read_text(encoding="utf-8").strip()
    if not raw:
      return {}
    return json.loads(raw)
  except Exception:
    return {}


def save_metadata(meta: Dict[str, Dict[str, Any]]) -> None:
  META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def file_stats_meta(path: Path, uploaded_at: Optional[str] = None, sha256: Optional[str] = None) -> Dict[str, Any]:
  stat = path.stat()
  updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
  return {
    "filename": path.name,
    "size_bytes": stat.st_size,
    "uploaded_at": uploaded_at,
    "updated_at": updated_at,
    "sha256": sha256,
    "pages": None,
  }


def update_pages_meta(page_counts: Dict[str, int]) -> None:
  if not page_counts:
    return
  meta = load_metadata()
  for filename, pages in page_counts.items():
    if filename not in meta:
      meta[filename] = {"filename": filename}
    meta[filename]["pages"] = pages
  save_metadata(meta)


def parse_chunk_meta(text: str) -> tuple[str, Optional[str], Optional[int]]:
  match = re.match(r"^\[source:\s*(.+?)\s*\|\s*page:\s*(\d+)\]\s*(.*)$", text, re.DOTALL)
  if not match:
    return text, None, None
  source = match.group(1).strip()
  page = int(match.group(2))
  clean_text = match.group(3).strip()
  return clean_text, source, page


def format_chunk_with_meta(filename: str, page: int, chunk: str) -> str:
  return f"[source: {filename} | page: {page}] {chunk}"


def sse_event(event: str, data: Any) -> str:
  payload = json.dumps(data, ensure_ascii=True)
  return f"event: {event}\ndata: {payload}\n\n"


def ensure_embed_model_loaded() -> None:
  global _EMBED_MODEL_READY
  if not _EMBED_MODEL_READY:
    rag_rust.load_embed_model()
    _EMBED_MODEL_READY = True


def embed_texts(texts: List[str]) -> List[List[float]]:
  """Embed indexed passages via Rust fastembed (BGE expects 'passage:' prefix)."""
  ensure_embed_model_loaded()
  prefixed_texts = [f"passage: {t}" for t in texts]
  return rag_rust.embed_texts_rust(prefixed_texts)


def embed_query(query: str) -> List[float]:
  """Embed query via Rust fastembed with BGE query prefix."""
  ensure_embed_model_loaded()
  prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
  return rag_rust.embed_texts_rust([prefixed_query])[0]


def load_pdf_pages_with_meta(max_pages: Optional[int]) -> List[Dict[str, Any]]:
  paths = list(PDF_DIR.glob("*.pdf"))
  if not paths:
    raise HTTPException(status_code=400, detail="No PDFs found in data/pdfs.")

  pages: List[Dict[str, Any]] = []
  page_counts: Dict[str, int] = {}

  for path in paths:
    try:
      file_pages = rag_rust.load_pdf_pages_many([str(path)])
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"Rust PDF loader failed: {exc}") from exc

    page_counts[path.name] = len(file_pages)
    for idx, text in enumerate(file_pages, start=1):
      if text and text.strip():
        pages.append({"filename": path.name, "page": idx, "text": text})
      if max_pages is not None and len(pages) >= max_pages:
        update_pages_meta(page_counts)
        return pages

  update_pages_meta(page_counts)
  return pages


def chunk_text(text: str) -> List[str]:
  if not text or not text.strip():
    return []
  return [
    chunk
    for chunk in rag_rust.smart_chunker(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if chunk
  ]


def build_index(rebuild: bool, max_pages: Optional[int]) -> dict:
  global _INDEX_READY
  pages = load_pdf_pages_with_meta(max_pages)

  display_chunks: List[str] = []
  embed_chunks: List[str] = []
  for page in pages:
    page_chunks = chunk_text(page["text"])
    for chunk in page_chunks:
      display_chunks.append(format_chunk_with_meta(page["filename"], page["page"], chunk))
      embed_chunks.append(chunk)

  if not display_chunks:
    raise HTTPException(status_code=400, detail="No text extracted from PDFs.")

  embeddings = embed_texts(embed_chunks)

  try:
    rag_rust.lancedb_create_or_open(
      DB_DIR,
      TABLE_NAME,
      display_chunks,
      embeddings,
      rebuild,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"LanceDB error: {exc}") from exc

  _INDEX_READY = True
  return {
    "pages": len(pages),
    "chunks": len(display_chunks),
    "rebuild": rebuild,
  }


def run_index(rebuild: bool, max_pages: Optional[int]) -> dict:
  global _INDEX_STATUS, _INDEX_INFO, _INDEX_READY
  _INDEX_STATUS = "building"
  _INDEX_READY = False
  _INDEX_INFO["last_error"] = None
  start = time.perf_counter()
  try:
    stats = build_index(rebuild, max_pages)
  except HTTPException as exc:
    _INDEX_STATUS = "error"
    _INDEX_INFO["last_error"] = exc.detail
    raise
  except Exception as exc:
    _INDEX_STATUS = "error"
    _INDEX_INFO["last_error"] = str(exc)
    raise
  end = time.perf_counter()
  build_ms = (end - start) * 1000
  stats["build_ms"] = build_ms
  _INDEX_INFO.update(
    {
      "last_build_at": utc_now_iso(),
      "last_build_ms": build_ms,
      "pages": stats.get("pages"),
      "chunks": stats.get("chunks"),
    }
  )
  _INDEX_STATUS = "ready"
  _INDEX_READY = True
  return stats


def ollama_chat(messages: List[dict], model_override: Optional[str]) -> tuple[str, Optional[str]]:
  payload = {
    "model": model_override or OLLAMA_CHAT_MODEL,
    "messages": messages,
    "stream": False,
    "options": {
      "temperature": CHAT_TEMPERATURE,
    },
  }
  resp = requests.post(
    f"{OLLAMA_BASE_URL}/api/chat",
    json=payload,
    timeout=OLLAMA_TIMEOUT,
  )
  if not resp.ok:
    raise HTTPException(status_code=resp.status_code, detail=resp.text)
  data = resp.json()
  message = data.get("message") or {}
  content = message.get("content", "")
  model_used = data.get("model") or payload["model"]
  return content, model_used


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/documents")
def upload_documents(files: List[UploadFile] = File(...)):
  ensure_dirs()
  meta = load_metadata()
  saved = []
  for f in files:
    if not f.filename.lower().endswith(".pdf"):
      raise HTTPException(status_code=400, detail=f"Not a PDF: {f.filename}")
    target = PDF_DIR / f.filename
    with target.open("wb") as out:
      file_bytes = f.file.read()
      out.write(file_bytes)
    saved.append(f.filename)
    meta[f.filename] = file_stats_meta(
      target,
      uploaded_at=utc_now_iso(),
      sha256=sha256_bytes(file_bytes),
    )
  save_metadata(meta)
  global _INDEX_READY, _INDEX_STATUS
  _INDEX_READY = False
  _INDEX_STATUS = "stale"
  return {"saved": saved, "needs_reindex": True}


@app.get("/documents")
def list_documents():
  ensure_dirs()
  meta = load_metadata()
  docs: List[Dict[str, Any]] = []
  files = sorted(PDF_DIR.glob("*.pdf"))
  for path in files:
    existing = meta.get(path.name, {})
    uploaded_at = existing.get("uploaded_at")
    sha256 = existing.get("sha256")
    if not uploaded_at or not sha256:
      meta[path.name] = file_stats_meta(path, uploaded_at=uploaded_at, sha256=sha256)
    else:
      meta[path.name].update(file_stats_meta(path, uploaded_at=uploaded_at, sha256=sha256))
    docs.append(meta[path.name])
  save_metadata(meta)
  return {"files": [p.name for p in files], "documents": docs}


def clear_index() -> None:
  global _INDEX_READY
  _INDEX_READY = False
  db_path = Path(DB_DIR)
  if db_path.exists():
    shutil.rmtree(db_path)
  global _INDEX_STATUS, _INDEX_INFO
  _INDEX_STATUS = "idle"
  _INDEX_INFO.update(
    {
      "pages": 0,
      "chunks": 0,
      "last_build_ms": None,
      "last_build_at": None,
      "last_error": None,
    }
  )


@app.delete("/documents/{filename}")
def delete_document(filename: str, rebuild_index: bool = True):
  ensure_dirs()
  target = PDF_DIR / filename
  if not target.exists():
    raise HTTPException(status_code=404, detail="File not found.")

  target.unlink()
  meta = load_metadata()
  if filename in meta:
    meta.pop(filename, None)
    save_metadata(meta)

  remaining = list(PDF_DIR.glob("*.pdf"))
  if not remaining:
    clear_index()
    return {"deleted": filename, "index_ready": False, "index_cleared": True}

  if rebuild_index:
    stats = run_index(rebuild=True, max_pages=None)
    return {"deleted": filename, "index_ready": True, "index": stats}

  _INDEX_READY = False
  global _INDEX_STATUS
  _INDEX_STATUS = "stale"
  return {"deleted": filename, "index_ready": False, "needs_reindex": True}


@app.post("/index")
def build_index_endpoint(payload: IndexRequest):
  return run_index(payload.rebuild, payload.max_pages)


@app.get("/index/status")
def index_status():
  return {
    "status": _INDEX_STATUS,
    "ready": _INDEX_READY,
    "info": _INDEX_INFO,
  }


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
  if not _INDEX_READY:
    raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

  def retrieve_chunks_with_meta(query: str, top_k: int) -> tuple[List[tuple], List[Dict[str, Any]]]:
    query_vector = embed_query(query)
    try:
      hits = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vector, top_k)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc
    chunks: List[tuple] = []
    meta: List[Dict[str, Any]] = []
    for text, dist in hits:
      clean_text, source, page = parse_chunk_meta(text)
      chunks.append((clean_text, dist))
      meta.append(
        {
          "text": clean_text,
          "distance": dist,
          "source": source,
          "page": page,
        }
      )
    return chunks, meta

  if not payload.use_llm:
    hits, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
    return QueryResponse(answer=None, model_used=None, retrieved=meta, mode="retrieval_only")

  if payload.mode.lower() == "naive":
    hits, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
    context_text = "\n".join([f"- {row['text']}" for row in meta])
    system_prompt = (
      "You are a helpful chatbot.\n"
      "Use only the following pieces of context to answer the question. "
      "Don't make up any new information:\n"
      f"{context_text}"
    )
    answer, model_used = ollama_chat(
      [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.question},
      ],
      payload.chat_model,
    )
    return QueryResponse(
      answer=answer,
      model_used=model_used,
      retrieved=meta,
      mode="naive",
      models={"generator": model_used},
    )

  state: Dict[str, Any] = {
    "query": payload.question,
    "attempts": 0,
    "should_retry": True,
    "trace": [] if payload.return_trace else None,
  }

  def retrieve_for_agent(q: str, top_k: int) -> List[tuple]:
    hits, meta = retrieve_chunks_with_meta(q, top_k)
    state["retrieved_meta"] = meta
    return hits

  refiner = QueryRefiner(lambda messages: ollama_chat(messages, payload.chat_model))
  retriever = Retriever(lambda q, top_k: retrieve_for_agent(q, top_k), top_k=payload.top_k)
  generator = Generator(lambda messages: ollama_chat(messages, payload.chat_model))
  evaluator = Evaluator(
    lambda messages: ollama_chat(messages, payload.chat_model),
    min_score=payload.min_score,
    max_attempts=payload.max_attempts,
  )
  proxy = UserProxy(refiner, retriever, generator, evaluator)
  state = proxy.run(state)

  retrieved = state.get("retrieved_meta") or []

  return QueryResponse(
    answer=state.get("answer"),
    model_used=state.get("model_used"),
    retrieved=retrieved,
    mode="agentic",
    refined_query=state.get("refined_query"),
    score=state.get("score"),
    attempts=state.get("attempts"),
    trace=state.get("trace") if payload.return_trace else None,
    models=state.get("models"),
  )


@app.post("/query/stream")
def query_stream(payload: QueryRequest):
  if not _INDEX_READY:
    raise HTTPException(status_code=400, detail="Index not built. Call /index first.")

  def retrieve_chunks_with_meta(query: str, top_k: int) -> tuple[List[tuple], List[Dict[str, Any]]]:
    query_vector = embed_query(query)
    try:
      hits = rag_rust.lancedb_search(DB_DIR, TABLE_NAME, query_vector, top_k)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"LanceDB search failed: {exc}") from exc
    chunks: List[tuple] = []
    meta: List[Dict[str, Any]] = []
    for text, dist in hits:
      clean_text, source, page = parse_chunk_meta(text)
      chunks.append((clean_text, dist))
      meta.append(
        {
          "text": clean_text,
          "distance": dist,
          "source": source,
          "page": page,
        }
      )
    return chunks, meta

  def event_stream() -> Iterable[str]:
    yield sse_event("status", {"state": "started"})

    if not payload.use_llm:
      _, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
      yield sse_event("retrieved", {"items": meta})
      yield sse_event("final", {"mode": "retrieval_only", "retrieved": meta})
      return

    if payload.mode.lower() == "naive":
      chunks, meta = retrieve_chunks_with_meta(payload.question, payload.top_k)
      yield sse_event("retrieved", {"items": meta})
      context_text = "\n".join([f"- {row['text']}" for row in meta])
      system_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n"
        f"{context_text}"
      )
      answer, model_used = ollama_chat(
        [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": payload.question},
        ],
        payload.chat_model,
      )
      yield sse_event("answer", {"answer": answer, "model_used": model_used})
      yield sse_event(
        "final",
        {
          "mode": "naive",
          "answer": answer,
          "model_used": model_used,
          "retrieved": meta,
          "models": {"generator": model_used},
        },
      )
      return

    pending: List[Dict[str, Any]] = []

    def emit(item: Dict[str, Any]) -> None:
      pending.append(item)

    state: Dict[str, Any] = {
      "query": payload.question,
      "attempts": 0,
      "should_retry": True,
      "trace": [] if payload.return_trace else None,
      "emit": emit,
    }

    def retrieve_for_agent(q: str, top_k: int) -> List[tuple]:
      hits, meta = retrieve_chunks_with_meta(q, top_k)
      state["retrieved_meta"] = meta
      return hits

    refiner = QueryRefiner(lambda messages: ollama_chat(messages, payload.chat_model))
    retriever = Retriever(lambda q, top_k: retrieve_for_agent(q, top_k), top_k=payload.top_k)
    generator = Generator(lambda messages: ollama_chat(messages, payload.chat_model))
    evaluator = Evaluator(
      lambda messages: ollama_chat(messages, payload.chat_model),
      min_score=payload.min_score,
      max_attempts=payload.max_attempts,
    )

    while state["should_retry"]:
      state = refiner.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()

      state = retriever.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()
      yield sse_event("retrieved", {"items": state.get("retrieved_meta") or []})

      state = generator.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()
      yield sse_event("answer", {"answer": state.get("answer"), "model_used": state.get("model_used")})

      state = evaluator.run(state)
      for item in pending:
        yield sse_event("trace", item)
      pending.clear()
      yield sse_event(
        "evaluation",
        {
          "score": state.get("score"),
          "summary": state.get("judge_summary"),
          "should_retry": state.get("should_retry"),
          "attempts": state.get("attempts"),
        },
      )

      if state.get("should_retry"):
        yield sse_event(
          "retry",
          {
            "attempts": state.get("attempts"),
            "score": state.get("score"),
          },
        )

    final_payload = QueryResponse(
      answer=state.get("answer"),
      model_used=state.get("model_used"),
      retrieved=state.get("retrieved_meta") or [],
      mode="agentic",
      refined_query=state.get("refined_query"),
      score=state.get("score"),
      attempts=state.get("attempts"),
      trace=state.get("trace") if payload.return_trace else None,
      models=state.get("models"),
    )
    yield sse_event("final", final_payload.model_dump())

  return StreamingResponse(event_stream(), media_type="text/event-stream")
