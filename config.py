"""Shared settings for the whole project.

Every LLM and embedding call from all three systems goes through the local
gateway (gateway.py).  The gateway forwards the call to the real model and
writes one log line per call, so we can measure the efficiency of RAG,
GraphRAG and LightRAG in exactly the same way.

    our code / GraphRAG / LightRAG  --->  gateway (localhost)  --->  Zhipu GLM  (chat)
                                                               --->  Ollama bge-m3 (embeddings)
"""
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Tokenizer file shipped with the project (tiktoken would otherwise download it,
# which can fail on some networks). All three systems use the same tokenizer.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "tiktoken_cache"))

# Local addresses must never go through a VPN / system proxy (this also applies to
# GraphRAG and LightRAG, which read the same setting).
_no_proxy = "127.0.0.1,localhost"
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = ",".join(x for x in [os.environ.get(_k, ""), _no_proxy] if x)
TOKENIZER = "cl100k_base"

# ---- models -----------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.7-flash")      # free Zhipu model
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")          # local Ollama model
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))           # bge-m3 vector size

# ---- chunking: identical for all three systems (fair comparison) -------------
CHUNK_TOKENS = 1200
CHUNK_OVERLAP = 100
TOP_K = 5                                                 # chunks given to the LLM in plain RAG

# ---- folders -------------------------------------------------------------------
DATA_TXT = ROOT / "data" / "txt"
RAG_DIR = ROOT / "rag_store"
GRAPHRAG_DIR = ROOT / "graphrag_proj"
LIGHTRAG_DIR = ROOT / "lightrag_store"
METRICS_DIR = ROOT / "metrics"
LOG_DIR = ROOT / "logs"
for d in (METRICS_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

# ---- gateway ---------------------------------------------------------------------
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8800"))
GATEWAY = f"http://127.0.0.1:{GATEWAY_PORT}"
GATEWAY_LOG = LOG_DIR / "gateway_calls.jsonl"


def base_url(tag: str) -> str:
    """OpenAI-style base URL for one 'tag' (e.g. rag_query, graphrag_index).

    The tag is part of the URL, so the gateway knows which system made each call.
    """
    return f"{GATEWAY}/{tag}/v1"


_gateway_thread = None


def _gateway_healthy() -> bool:
    try:
        return httpx.get(f"{GATEWAY}/health", timeout=2, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_gateway(timeout: float = 30) -> None:
    """Start the gateway inside this Python process (background thread) if none is running.

    It lives exactly as long as the script that needs it, so there is no
    left-over background program to manage.
    """
    global _gateway_thread
    if _gateway_healthy():
        return
    import threading
    import uvicorn
    import gateway
    server = uvicorn.Server(uvicorn.Config(gateway.app, host="127.0.0.1", port=GATEWAY_PORT,
                                           log_level="warning"))
    _gateway_thread = threading.Thread(target=server.run, daemon=True, name="gateway")
    _gateway_thread.start()
    start = time.time()
    while time.time() - start < timeout:
        if _gateway_healthy():
            return
        time.sleep(0.3)
    raise RuntimeError(f"The local gateway could not start on port {GATEWAY_PORT}")


def client(tag: str) -> OpenAI:
    ensure_gateway()
    return OpenAI(api_key="via-gateway", base_url=base_url(tag), timeout=600, max_retries=0,
                  http_client=httpx.Client(trust_env=False, timeout=600))


def chat(messages, tag="rag_query", temperature=0.1, max_tokens=1500) -> str:
    """One chat completion through the gateway; returns the answer text."""
    r = client(tag).chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return (r.choices[0].message.content or "").strip()


def embed(texts, tag="rag_query"):
    """Embed a list of texts with bge-m3 through the gateway."""
    r = client(tag).embeddings.create(model=EMBED_MODEL, input=list(texts))
    return [d.embedding for d in r.data]


def folder_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6, 2)


def graphrag_env(tag: str) -> dict:
    """Environment variables that fill the <dollar>{...} values in graphrag_proj/settings.yaml."""
    env = os.environ.copy()
    env.update({
        "GRAPHRAG_LLM_MODEL": LLM_MODEL,
        "GRAPHRAG_EMBED_MODEL": EMBED_MODEL,
        "GRAPHRAG_LLM_BASE": base_url(tag),
        "GRAPHRAG_API_KEY": "via-gateway",
        "GRAPHRAG_METRICS_DIR": str(METRICS_DIR / "graphrag_llm" / tag),
        "PYTHONIOENCODING": "utf-8",
    })
    return env
