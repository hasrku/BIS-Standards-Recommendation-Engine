import json
import pickle
import faiss
import re
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text):
    """Smarter tokenizer that ignores punctuation like commas and periods."""
    return re.findall(r'\w+', text.lower())


print("Loading data...")
with open(os.path.join("data", "bis_chunks.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
texts = [d["content"] for d in data]

print("Building BM25 Keyword Index...")
bm25 = BM25Okapi([tokenize(t) for t in texts])
with open(os.path.join("data", "bm25.pkl"), "wb") as f:
    pickle.dump(bm25, f)

print("Building FAISS Vector Index...")
# We can use your local embedder now!
embedder = SentenceTransformer('./local_models/embedder')
idx = faiss.IndexFlatL2(384)
idx.add(embedder.encode(texts))
faiss.write_index(idx, os.path.join("data", "vector.index"))

print("Indexing Complete!")
