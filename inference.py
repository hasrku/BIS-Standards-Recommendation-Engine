import json
import argparse
import time
import pickle
import faiss
import numpy as np
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------
# 1. LOAD PRE-COMPUTED DATA & MODELS (Loads once at startup)
# ---------------------------------------------------------
print("Loading Models and Indexes... (This doesn't count towards latency)")

# Path to the data directory
DATA_DIR = "data"
MODELS_DIR = "local_models"

# Load chunks
with open(os.path.join(DATA_DIR, "bis_chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load BM25
with open(os.path.join(DATA_DIR, "bm25.pkl"), "rb") as f:
    bm25 = pickle.load(f)

# Load FAISS & Embedding Model from local folders
embedder = SentenceTransformer(os.path.join(MODELS_DIR, 'embedder'))
vector_index = faiss.read_index(os.path.join(DATA_DIR, "vector.index"))

# Load Reranker from local folder (The secret weapon for high MRR)
reranker = CrossEncoder(os.path.join(MODELS_DIR, 'reranker'))


def get_hybrid_top_k(query, k=30):
    """Retrieves top K chunks using both BM25 and Vector Search."""
    # --- BM25 Search ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]

    # --- Vector Search ---
    query_vector = embedder.encode([query])
    _, faiss_top_indices = vector_index.search(query_vector, k)

    # Combine unique indices
    combined_indices = set(bm25_top_indices).union(set(faiss_top_indices[0]))
    return list(combined_indices)


def process_queries(input_file, output_file):
    with open(input_file, 'r') as f:
        queries = json.load(f)

    results = []

    print(f"Processing {len(queries)} queries...")
    for item in queries:
        start_time = time.time()
        user_query = item["query"]

        # 1. Hybrid Retrieval (Get top 15-20 candidates)
        candidate_indices = get_hybrid_top_k(user_query, k=15)

        # 2. Reranking (Score query against each candidate chunk directly)
        cross_inp = [[user_query, chunks[idx]["content"]]
                     for idx in candidate_indices]
        cross_scores = reranker.predict(cross_inp)

        # Sort candidates by the reranker's highly accurate score
        scored_candidates = list(zip(candidate_indices, cross_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Extract Top 5 Standard IDs (Zero Hallucination Trick)
        top_5_standards = []
        seen_ids = set()

        for idx, score in scored_candidates:
            std_id = chunks[idx]["standard_id"]
            if std_id not in seen_ids:
                top_5_standards.append(std_id)
                seen_ids.add(std_id)
            if len(top_5_standards) == 5:
                break

        # 4. (Optional for Demo) LLM Rationale Generation
        # Here is where you would call an API or local LLM like Ollama
        # rationale = llm.generate(f"Query: {user_query}\nStandard: {top_5_standards[0]}")

        latency = time.time() - start_time

        # 5. Save exactly to the Hackathon Schema
        results.append({
            "id": item["id"],
            "query": user_query,
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": top_5_standards,
            "latency_seconds": round(latency, 3)
        })

        print(
            f"Processed {item['id']} in {latency:.2f}s | Top Match: {top_5_standards[0] if top_5_standards else 'None'}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to hidden_private_dataset.json")
    parser.add_argument("--output", required=True,
                        help="Path to save team_results.json")
    args = parser.parse_args()

    process_queries(args.input, args.output)
