import json
import pickle
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

data = json.load(open("bis_chunks.json"))
texts = [d["content"] for d in data]

pickle.dump(BM25Okapi([t.lower().split()
            for t in texts]), open("bm25.pkl", "wb"))

model = SentenceTransformer('all-MiniLM-L6-v2')
idx = faiss.IndexFlatL2(384)
idx.add(model.encode(texts))
faiss.write_index(idx, "vector.index")
