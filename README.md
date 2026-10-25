# RAG-based Dialogue System over Airbus and Boeing Documents

**HW2 – Task One** · Umair Muhammad (2024380178) · School of Computer Science, Northwestern Polytechnical University

A dialogue system that answers questions about official Airbus and Boeing documents, built with three
knowledge-augmentation methods on exactly the same data and models, and a measured comparison of them:

| Method | Library | Link from the homework |
|---|---|---|
| Plain RAG | FAISS + bge-m3 (own implementation) | – |
| GraphRAG | Microsoft GraphRAG `graphrag==3.3.0` | https://graphrag.homes/ |
| LightRAG | HKUDS LightRAG `lightrag-hku==1.5.7` | https://github.com/HKUDS/LightRAG |

The full report (7 chapters, ACM single-column format) is in [`report/HW2_2024380178.pdf`](report/HW2_2024380178.pdf).

## Results (30 test questions, LLM-judge score 1–5)

| Engine | Factual | Multi-hop | Global | Unanswerable | **All** | Tokens / question |
|---|---|---|---|---|---|---|
| Plain RAG | 4.27 | 3.29 | 3.60 | 5.00 | **4.00** | 6,342 |
| GraphRAG (local) | 4.80 | 3.86 | 4.20 | 5.00 | **4.50** | 11,643 |
| GraphRAG (global) | 3.80 | 3.29 | 4.20 | 5.00 | **3.87** | 107,064 |
| LightRAG (hybrid) | 4.93 | 3.14 | 4.20 | 5.00 | **4.40** | 30,724 |

| Index | LLM calls | LLM tokens | Entities | Relations | Community reports |
|---|---|---|---|---|---|
| Plain RAG | 0 | 0 | – | – | – |
| GraphRAG | 742 | 1,798,394 | 1,416 | 2,658 | 261 |
| LightRAG | 115 | 508,054 | 1,471 | 1,392 | – |

All raw results are included: `eval/results.csv` (every answer and score), `eval/dialogue_test.csv`,
`metrics/*.json` and `logs/gateway_calls.jsonl` (one line per model call).

## Data
Six official documents (417 pages, 49,865 words after cleaning):

| Maker | Document |
|---|---|
| Airbus | A320 Aircraft Characteristics – Airport and Maintenance Planning (Rev. Jun 2024) |
| Airbus | Safety First magazine, issues 23 and 25 |
| Boeing | 737 MAX Airplane Characteristics for Airport Planning (D6-38A004, Rev. K) |
| Boeing | 787 Airplane Characteristics for Airport Planning (D6-58333, Rev. Q) |
| Boeing | Statistical Summary of Commercial Jet Airplane Accidents |

The PDFs are not redistributed here; the cleaned text used by all three systems is in `data/txt/`
(page ranges in `corpus.json`).

## Run the chatbot with the pre-built indexes (no re-indexing needed)

Requirements: Windows 10/11, **Python 3.12**, **[Ollama](https://ollama.com/download)**, and a free API key
from **[Zhipu BigModel](https://open.bigmodel.cn)** (the free GLM-Flash models are used).

1. Download this repository (green **Code** button → *Download ZIP*) and unzip it.
2. Download **`prebuilt_indexes.zip`** from the [Releases](../../releases) page of this repository and unzip it
   **into the same folder** (it creates `rag_store/`, `lightrag_store/` and `graphrag_proj/`).
3. Install the embedding model: open a terminal and run `ollama pull bge-m3`.
4. Copy `env.example.txt` to a new file named `.env` and put your own Zhipu API key in it.
5. In PowerShell, inside the folder:
   ```
   powershell -ExecutionPolicy Bypass -File setup.ps1     # once: creates .venv and installs the packages
   .\.venv\Scripts\python.exe app.py                      # opens the chat at http://127.0.0.1:7860
   ```
6. Choose the engine (Plain RAG, GraphRAG local/global, LightRAG hybrid/mix) under *Additional Inputs* and ask,
   for example, *"What is the maximum design takeoff weight of the Boeing 787-8?"* and then *"And the 787-9?"*.

`python app.py --share` additionally prints a temporary public `gradio.live` link.

## Rebuild everything from the PDFs
Put the six PDFs into `data/pdf/` (file names as in `corpus.json`) and run
```
powershell -ExecutionPolicy Bypass -File run_all.ps1
```
This prepares the text, builds the three indexes, evaluates the engines and regenerates the report tables and
figures. Finished steps are skipped, so the script can be restarted at any time. On the free API tier the graph
indexes take several hours (see Chapter 6 of the report).

## Code overview

| File | Purpose |
|---|---|
| `corpus.json`, `01_prepare_data.py` | PDF → clean text (tables kept as Markdown rows) |
| `02_build_rag.py` | plain RAG: 1200-token chunks → bge-m3 → FAISS |
| `03_build_graphrag.py`, `configs/graphrag_settings.yaml` | GraphRAG index (`graphrag init` / `graphrag index`) |
| `04_build_lightrag.py`, `lightrag_setup.py` | LightRAG index |
| `engines.py` | the query engines behind one interface + follow-up question rewriting |
| `app.py` | Gradio dialogue system |
| `05_evaluate.py`, `eval/questions.csv` | 30 questions with references, LLM judge, dialogue test |
| `06_make_report_assets.py` | tables, figures and numbers for the report |
| `gateway.py`, `config.py`, `configs/llm_pool.txt` | local gateway: model pool, retries, time-outs, logging of every call |
| `run_all.ps1`, `setup.ps1` | Windows scripts for setup and the full pipeline |

All three systems use the same chunking (1200 tokens, 100 overlap, `cl100k_base`), the same LLMs and the same
embedding model, so differences in the results come from the retrieval method.
