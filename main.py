import json
import pickle
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR = "data"
MODELS_DIR = "local_models"

# ─────────────────────────────────────────────
# Global state (populated once at startup)
# ─────────────────────────────────────────────
state: dict = {}

# Thread pool — offloads CPU-heavy inference off the async event loop
# so FastAPI stays responsive and latency matches the CLI baseline.
_executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────
# Lifespan: load everything ONCE at startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("═" * 55)
    print("  BIS × IITT  —  Loading models & indexes …")
    print("═" * 55)

    with open(os.path.join(DATA_DIR, "bis_chunks.json"), "r", encoding="utf-8") as f:
        state["chunks"] = json.load(f)
    print(f"  ✓ Loaded {len(state['chunks'])} chunks")

    with open(os.path.join(DATA_DIR, "bm25.pkl"), "rb") as f:
        state["bm25"] = pickle.load(f)
    print("  ✓ Loaded BM25 index")

    state["embedder"] = SentenceTransformer(
        os.path.join(MODELS_DIR, "embedder"))
    state["vector_index"] = faiss.read_index(
        os.path.join(DATA_DIR, "vector.index"))
    print("  ✓ Loaded FAISS index + embedder")

    state["reranker"] = CrossEncoder(os.path.join(MODELS_DIR, "reranker"))
    print("  ✓ Loaded reranker")

    print("═" * 55)
    print("  All models ready. Server is live.")
    print("═" * 55)

    yield  # server runs here

    state.clear()
    print("Server shutting down — state cleared.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="BIS × IITT Retrieval API", lifespan=lifespan)


# ─────────────────────────────────────────────
# Core retrieval helpers  (pure sync — run in executor)
# ─────────────────────────────────────────────

def _hybrid_retrieve(query: str, k: int = 15) -> list:
    """BM25 ∪ FAISS candidate retrieval."""
    bm25: BM25Okapi = state["bm25"]
    embedder: SentenceTransformer = state["embedder"]
    vector_index = state["vector_index"]

    tokenized = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized)
    bm25_top = np.argsort(bm25_scores)[::-1][:k].tolist()

    qvec = embedder.encode([query])
    _, faiss_top = vector_index.search(qvec, k)

    return list(set(bm25_top) | set(faiss_top[0].tolist()))


def _rerank_and_select(query: str, candidate_indices: list, top_n: int = 5) -> list:
    """Cross-encoder reranking → top_n unique standards."""
    chunks = state["chunks"]
    reranker: CrossEncoder = state["reranker"]

    cross_inp = [[query, chunks[idx]["content"]] for idx in candidate_indices]
    cross_scores = reranker.predict(cross_inp)

    scored = sorted(zip(candidate_indices, cross_scores),
                    key=lambda x: x[1], reverse=True)

    results, seen_ids = [], set()
    for idx, score in scored:
        std_id = chunks[idx]["standard_id"]
        if std_id not in seen_ids:
            seen_ids.add(std_id)
            results.append({
                "rank": len(results) + 1,
                "standard_id": std_id,
                "score": round(float(score), 4),
                "snippet": chunks[idx]["content"][:300].strip(),
            })
        if len(results) == top_n:
            break

    return results


def _run_pipeline_sync(query: str) -> dict:
    """
    Synchronous full pipeline.
    Called via run_in_executor so it never blocks the event loop.
    """
    t0 = time.time()
    candidates = _hybrid_retrieve(query, k=15)
    ranked = _rerank_and_select(query, candidates, top_n=5)
    latency = round(time.time() - t0, 3)
    return {"query": query, "results": ranked, "latency_seconds": latency}


async def run_pipeline(query: str) -> dict:
    """Async wrapper — offloads CPU work to thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_pipeline_sync, query)


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def compute_metrics(batch_results: list) -> dict:
    """
    Compute Hit Rate@3, MRR@5, and Avg Latency.
    Items need: expected_standards, retrieved_standards, latency_seconds.
    """
    mrr_total = 0.0
    hit3_total = 0
    latency_total = 0.0
    n = 0

    for item in batch_results:
        expected = set(item.get("expected_standards", []))
        retrieved = item.get("retrieved_standards", [])
        latency_total += float(item.get("latency_seconds", 0))

        if not expected:
            continue  # skip items without ground truth for accuracy metrics

        n += 1

        # MRR@5 — rank of first relevant result (up to position 5)
        for rank, std_id in enumerate(retrieved[:5], start=1):
            if std_id in expected:
                mrr_total += 1.0 / rank
                break

        # Hit Rate@3 — at least one relevant in top-3
        if set(retrieved[:3]) & expected:
            hit3_total += 1

    total = len(batch_results)
    avg_latency = round(latency_total / total, 3) if total else 0.0

    if n == 0:
        return {"avg_latency_seconds": avg_latency, "total_queries": total}

    return {
        "total_queries": total,
        "hit_rate_at_3": round(hit3_total / n, 4),
        "mrr_at_5": round(mrr_total / n, 4),
        "avg_latency_seconds": avg_latency,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("src/templates/index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(state)}


# ── Single Query ──────────────────────────────
class QueryRequest(BaseModel):
    query: str


@app.post("/api/query")
async def single_query(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    output = await run_pipeline(body.query.strip())
    return JSONResponse(content=output)


# ── Batch from JSON file ──────────────────────
@app.post("/api/batch")
async def batch_query(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400, detail="Please upload a .json file.")
    if not state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    raw = await file.read()
    try:
        queries = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    if not isinstance(queries, list):
        raise HTTPException(
            status_code=400, detail="JSON must be a list of query objects.")

    results = []
    for item in queries:
        query_text = item.get("query", "").strip()
        if not query_text:
            continue

        pipeline_out = await run_pipeline(query_text)
        retrieved_standards = [r["standard_id"]
                               for r in pipeline_out["results"]]

        results.append({
            "id": item.get("id", len(results) + 1),
            "query": query_text,
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": retrieved_standards,
            "ranked_results": pipeline_out["results"],
            "latency_seconds": pipeline_out["latency_seconds"],
        })

    return JSONResponse(content={"results": results, "total": len(results)})


# ── Evaluate batch results ────────────────────
class EvaluateRequest(BaseModel):
    results: List[dict]


@app.post("/api/evaluate")
async def evaluate(body: EvaluateRequest):
    if not body.results:
        raise HTTPException(status_code=400, detail="No results provided.")
    metrics = compute_metrics(body.results)
    return JSONResponse(content=metrics)
