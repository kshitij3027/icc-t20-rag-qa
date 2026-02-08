# ICC T20 RAG Q&A System

A **Retrieval-Augmented Generation (RAG)** pipeline for answering questions about ICC T20 International cricket rules. Built as part of Week 2 of the AI Engineering Bootcamp.

The system loads ICC T20 rule PDFs, chunks them, embeds them with OpenAI, stores them in ChromaDB, and answers questions using a LangChain RAG chain powered by GPT-4o-mini—with retrieval testing, evaluation, and a chunk-size comparison experiment.

---

## Project Overview

| Component | Technology |
|-----------|------------|
| **Documents** | 4 ICC T20 rule PDFs (match structure, bowling/no-balls/free hits, DRS/umpiring, special situations) |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` (configurable chunk size/overlap) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector store** | ChromaDB (in-memory) |
| **LLM** | OpenAI GPT-4o-mini via LangChain |

### Pipeline Steps

1. **Load** — Read all 4 PDFs using LangChain's `PyPDFLoader`.
2. **Chunk** — Split with `RecursiveCharacterTextSplitter`; notebook compares 500 vs 1000 (and later 500/1000/1500 for the stretch goal).
3. **Embed + Store** — Embed chunks with OpenAI, store in ChromaDB with `source` and `page` metadata.
4. **Test Retrieval** — Run 3 test queries with `k=3` to verify relevant chunks are retrieved before adding the LLM.
5. **Build RAG Chain** — Custom prompt (answer only from context, cite sources, say "I don't have enough information" when appropriate) + `RetrievalQA` with GPT-4o-mini.
6. **Evaluate** — Run 5 questions from `eval_questions.json`; measure retrieval accuracy, faithfulness, and correctness.
7. **Stretch Goal A** — Compare retrieval/faithfulness/correctness across chunk sizes 500, 1000, and 1500.

---

## Repository Structure

```
.
├── README.md
├── ICC_T20_RAG_QA_System.ipynb   # Main Colab/local notebook
├── cricket_rules_data/
│   ├── 01_match_structure_and_playing_conditions.pdf
│   ├── 02_bowling_rules_no_balls_and_free_hits.pdf
│   ├── 03_drs_and_umpiring.pdf
│   ├── 04_special_match_situations.pdf
│   └── eval_questions.json       # 5 evaluation Q&A with expected answers & source docs
└── .gitignore
```

---

## Prerequisites

- **Python** 3.9+
- **OpenAI API key** (for embeddings and GPT-4o-mini)
- **Google account** (optional, only if running on Colab with Drive)

### Python packages (installed in the notebook)

- `langchain` (pre-1.0 for compatibility), `langchain-openai`, `langchain-chroma`, `langchain-community`, `langchain-text-splitters`
- `chromadb`, `pypdf`

---

## How to Clone and Run

### Option A: Run on Google Colab (recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/kshitij3027/icc-t20-rag-qa.git
   cd icc-t20-rag-qa
   ```

2. **Upload the data to Google Drive**
   - Create a folder in your Drive, e.g. `cricket_rules_data`.
   - Upload the 4 PDFs and `eval_questions.json` from `cricket_rules_data/` into that folder.

3. **Open the notebook in Colab**
   - Upload `ICC_T20_RAG_QA_System.ipynb` to Colab, or open it from Drive if you clone/copy the repo there.
   - Run the first cell to install dependencies.
   - Run the Drive mount cell and set:
     ```python
     DATA_PATH = "/content/drive/MyDrive/cricket_rules_data"
     ```
   - In Colab: **Secrets** (key icon) → add `OPENAI_API_KEY` with your OpenAI API key. The notebook uses `userdata.get('OPENAI_API_KEY')`.

4. **Run all cells** in order. The notebook will load, chunk, embed, build the RAG chain, run test queries, evaluate on the 5 questions, and run the chunk-size comparison.

### Option B: Run locally (Jupyter or VS Code)

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/kshitij3027/icc-t20-rag-qa.git
   cd icc-t20-rag-qa
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install "langchain<1.0" langchain-openai langchain-chroma langchain-community langchain-text-splitters chromadb pypdf
   ```

3. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   For the notebook to use it locally, you’ll need to change the API key cell: instead of Colab’s `userdata.get('OPENAI_API_KEY')`, use:
   ```python
   import os
   api_key = os.environ.get("OPENAI_API_KEY")
   if not api_key:
       raise ValueError("Set OPENAI_API_KEY environment variable")
   ```

4. **Point the notebook at the local data**
   - In the “Set Data Path” cell, set:
     ```python
     DATA_PATH = "./cricket_rules_data"   # or the full path to cricket_rules_data
     ```
   - Remove or skip the “Mount Google Drive” cell when running locally.

5. **Open and run the notebook**
   ```bash
   jupyter notebook ICC_T20_RAG_QA_System.ipynb
   ```
   Run all cells in order.

---

## Data

- **Knowledge base:** 4 PDFs covering T20I match structure, powerplay, bowling/no-balls/free hits, DRS/umpiring, and special situations (Super Over, DLS, etc.).
- **Evaluation:** `eval_questions.json` has 5 questions with `question`, `expected_answer`, `source_document`, and `source_section` for retrieval and correctness checks.

---

## Learnings & Takeaways

### What worked well

- **RAG pipeline:** Retrieval + GPT-4o-mini produced grounded, accurate answers for ICC T20 rules when the right chunks were retrieved.
- **ChromaDB + OpenAI embeddings:** `text-embedding-3-small` gave good semantic search over the rule documents.
- **Prompt design:** Instructing the model to answer only from context, cite the document, and say “I don’t have enough information” when appropriate reduced hallucination.
- **Chunk size:** For this rule-based corpus, **chunk_size=1000** with overlap 100 was a good balance: enough context per chunk without too much noise.

### Chunk size experiment (Stretch Goal A)

- **500:** More chunks, but rules were sometimes split across chunks, hurting retrieval and answer quality.
- **1000:** Best trade-off: full rules often fit in one chunk, retrieval and correctness were strong.
- **1500:** Fewer, larger chunks; sometimes multiple rules in one chunk added noise and diluted answers.

So for structured, paragraph-style rule text, a medium chunk size (e.g. 1000) worked better than very small or very large.

### What could be improved

- **Evaluation scale:** 5 questions is minimal; production would need 50–100+ diverse questions.
- **Retrieval:** Try different `k`, hybrid search (keyword + semantic), or re-ranking.
- **Embeddings:** Test `text-embedding-3-large` or other models.
- **Metadata filtering:** Pre-filter by document/section before similarity search.
- **Evaluation method:** Use a separate evaluator model or human eval instead of self-evaluation for more reliable scores.

### Overall

RAG is a practical and effective way to build a Q&A system over domain-specific documents like ICC T20 rules: load → chunk → embed → retrieve → generate with a constrained prompt.

---

## License

This project is for educational use as part of the AI Engineering Bootcamp.
