# BIS Standards Recommendation Engine

![Hit Rate @3](https://img.shields.io/badge/Hit_Rate_@3-100%25-success)
![MRR @5](https://img.shields.io/badge/MRR_@5-0.95-success)
![Avg Latency](https://img.shields.io/badge/Latency-1.45s-success)

_Note: used the input data : `public_test_set.json`._

This repository contains my submission for the **BIS Standards Recommendation Engine Hackathon**. I have built an offline-capable, blazing-fast Retrieval-Augmented Generation (RAG) pipeline designed to instantly map product descriptions from Micro and Small Enterprises to the correct Bureau of Indian Standards (BIS) regulations.

---

## System Architecture

To achieve sub-second latency with near-perfect accuracy, I bypassed slow LLM generation for standard extraction and implemented a **Dual-Retrieval + Reranking Architecture**:
![System Architecture](src/images/system_architecture.png)

1. **Smart Chunking:**
   I utilized custom Regex (with `IGNORECASE`) to intelligently slice the BIS SP 21 document into distinct, highly focused "flashcards."
2. **Dual-Retrieval (Casting a Wide Net, `k=15`):**
    - **Semantic Search:** `FAISS` combined with `all-MiniLM-L6-v2` catches conceptual matches (e.g., "water mains" -> pipes).
    - **Keyword Search:** A custom punctuation-stripped `BM25` implementation ensures exact matches (like specific grades or keywords) are never missed.
3. **Cross-Encoder Reranking (The Precision Judge):**
   The combined pool of 15 candidates is fed into the lightning-fast `ms-marco-MiniLM-L-6-v2` Cross-Encoder. This model scores the query directly against the text of the chunks.
4. **Deterministic Extraction:**
   Instead of asking an LLM to type out the standard (risking typos), our system uses pure Python logic to extract the `standard_id` directly from the metadata of the reranked chunks.

---

## 💻 Hardware & Environment Requirements

This pipeline is highly optimized for offline, local execution. It does not require expensive cloud infrastructure or high-end GPUs to achieve sub-second latency.

-   **Processor (CPU):** Any modern multi-core consumer CPU (Intel i3 / AMD Ryzen 3 or newer).
-   **Graphics (GPU):** Not required. The architecture utilizes `faiss-cpu` and lightweight Hugging Face models.
-   **Memory (RAM):** 4 GB minimum.
-   **Storage:** ~1.2 GB .
-   **OS:** Windows, macOS, or Linux.

---

## 📂 Repository Structure

According to the hackathon guidelines, the repository is structured as follows:

```
├── /data                  # Processed JSON chunks, Vector/BM25 indexes, and dataset files
├── /local_models          # Cached HuggingFace models for 100% offline execution
│   ├── /embedder          # sentence-transformers/all-MiniLM-L6-v2
│   └── /reranker          # cross-encoder/ms-marco-MiniLM-L-6-v2
├── /src                   # Application logic and UI assets
│   ├── /images            # Architecture and pipeline diagrams
│   ├── /templates         # HTML templates for the web interface
│   ├── chunker.py         # Custom regex-based intelligent document chunking
│   └── indexer.py         # FAISS and BM25 index generation script
├── eval_script.py         # Mandatory evaluation script provided by organizers
├── inference.py           # Mandatory entry-point script for judges (CLI)
├── main.py                # Web UI application logic
├── presentation.pdf       # 8-slide presentation deck
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Data Ingestion Strategy Disclosure

The provided `dataset.pdf` (BIS SP 21) was challenging to parse algorithmically due to its complex, multi-column tables and OCR irregularities.

To maintain momentum and ensure high-fidelity text extraction without hitting third-party API rate limits, I utilized **MinerU** ([mineru.net](https://mineru.net)), an open-source document understanding tool. I processed the PDF through MinerU to generate the foundational `dataset.md` file.

_Note: As this was a one-time preprocessing step, the RAG pipeline begins its automated execution directly from the `data/dataset.md` file._

---

## 💻 Setup & Execution Guide

1. Clone the Repo

```
git clone https://github.com/hasrku/BIS-Standards-Recommendation-Engine
```

2. Install Dependencies

```
python -m venv venv
```

3. Run the fastapi server

```
venv/Scripts/activate
```

4. Install Dependencies

```
pip install -r requirements.txt
```

5. Run the fastapi server

```
uvicorn main:app --reload
```

6. Open the url in the browser

```
http://localhost:8000
```

7. Run Command Line Inference Pipeline

```
python inference.py --input public_test_set.json --output team_results.json
```

8. Evaluate Results

```
python eval_script.py --results team_results.json
```

---

## 🚀 Usage & Input Data Format

The recommendation engine can be accessed via two primary interfaces: a user-friendly Web UI and a high-throughput Command Line Interface (CLI).

### 1. Web Interface (FastAPI)

Run the application and navigate to **`http://localhost:8000`** in your browser. The web dashboard offers two modes of operation:

-   **Quick Text Input:** Perfect for MSEs needing instant answers. You can enter a single product description or input multiple queries at once by separating them with a semicolon (`;`).
    -   _Example:_ `33 Grade Ordinary Portland Cement ; precast concrete pipes`
-   **Batch Query & Evaluation:** Designed for bulk processing and testing. Upload a JSON file formatted according to the hackathon's test set schema. The system will process all queries, evaluate the accuracy against the expected standards, and output the performance metrics.

### 2. Command Line Interface (CLI)

For automated testing and evaluation, you can run the batch pipeline directly from the terminal.

Place your input JSON file in the root directory of the project and execute the inference script:

```bash
python inference.py --input public_test_set.json --output team_results.json
```

### 📄 Batch Input JSON Schema

Whether you are uploading a file via the Web UI or running the CLI, your input JSON must strictly follow the format of the official `public_test_set.json`.

It must be an array of objects containing an `id`, the `query` (product description), and the `expected_standards` (for evaluation scoring).

**Input Format Skeleton:**

```json
[
    {
        "id": "String (e.g., 'PUB-01')",
        "query": "String (Product description goes here...)",
        "expected_standards": ["String (e.g., 'IS 269: 1989')"]
    }
    // ... additional query objects follow the exact same structure
]
```

---

## ▶️ Project Demo video

🎥 Watch the 7-Minute Demo Video [Here](https://drive.google.com/file/d/1MPUlSn97LUw2QUsz_cjyfgsf3Hp3hvW3/view?usp=sharing).
