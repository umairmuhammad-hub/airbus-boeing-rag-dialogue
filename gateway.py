"""Local LLM gateway - one place where every model call passes through.

Why it exists
-------------
1. Fair measurement: RAG, GraphRAG and LightRAG all send their calls here, so the
   number of LLM calls, tokens and seconds is counted in exactly the same way for
   each system (logs/gateway_calls.jsonl).
2. Free-tier friendly: it limits how many requests run at the same time and retries
   automatically when the free API says "too many requests" (HTTP 429).
3. Compatibility: GraphRAG asks for "json_schema" structured output, which the Zhipu
   API does not list; the gateway turns it into the supported "json_object" mode
   and adds the schema to the prompt.

URL scheme:  http://127.0.0.1:8800/<tag>/v1/chat/completions   (tag = who is calling)
             http://127.0.0.1:8800/<tag>/v1/embeddings
"""
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

LLM_URL = os.getenv("LLM_UPSTREAM_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
LLM_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.7-flash")
EMBED_URL = os.getenv("EMBED_UPSTREAM_URL", "http://127.0.0.1:11434/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
# Model pool: the same free Zhipu key serves two free models, each with its own small
# concurrency limit. Every request goes to whichever model has a free slot, which roughly
# doubles throughput on the free tier. Format: "model:slots,model:slots".
_pool_default = f"{LLM_MODEL}:1,glm-4-flash-250414:2" if "bigmodel" in LLM_URL else f"{LLM_MODEL}:1"
# configs/llm_pool.txt (if present) overrides this, e.g. "glm-4.5-air:8" for a paid model.
_pool_file = ROOT / "configs" / "llm_pool.txt"
_pool_text = os.getenv("LLM_POOL") or (_pool_file.read_text(encoding="utf-8").strip()
                                        if _pool_file.exists() else "") or _pool_default
LLM_POOL = [(m.split(":")[0].strip(), int((m.split(":") + ["1"])[1]))
            for m in _pool_text.split(",") if m.strip() and not m.strip().startswith("#")]
MAX_EMBED = int(os.getenv("EMBED_MAX_CONCURRENCY", "2"))
MAX_OUTPUT = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))  # stops runaway, repeating answers
DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "true").lower() == "true"
# Extra request fields that switch off "thinking" mode differ per provider:
if os.getenv("LLM_EXTRA_BODY"):
    EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY"))
elif "bigmodel" in LLM_URL:
    EXTRA_BODY = {"thinking": {"type": "disabled"}}        # Zhipu GLM
elif "siliconflow" in LLM_URL:
    EXTRA_BODY = {"enable_thinking": False}               # SiliconFlow Qwen3
else:
    EXTRA_BODY = {}                                       # e.g. DeepSeek non-thinking model
PORT = int(os.getenv("GATEWAY_PORT", "8800"))
LOG = ROOT / "logs" / "gateway_calls.jsonl"
LOG.parent.mkdir(exist_ok=True)

# Only these fields are forwarded (the Zhipu API rejects some OpenAI-only fields).
ALLOWED = {"model", "messages", "temperature", "top_p", "max_tokens", "stream", "stop",
           "tools", "tool_choice", "response_format", "thinking", "do_sample", "user_id"}

app = FastAPI()


class ModelPool:
    """Hands out one free (model) slot at a time; waits when all slots are busy."""

    def __init__(self, pool):
        self.free = {m: n for m, n in pool}
        self.order = [m for m, _ in pool]
        self.cond = None

    async def acquire(self, avoid=None):
        if self.cond is None:
            self.cond = asyncio.Condition()
        async with self.cond:
            while True:
                choices = [m for m in self.order if self.free[m] > 0]
                if avoid in choices and len(choices) > 1:
                    choices.remove(avoid)          # after a rate limit, prefer the other model
                if choices:
                    self.free[choices[0]] -= 1
                    return choices[0]
                await self.cond.wait()

    async def release(self, model):
        async with self.cond:
            self.free[model] += 1
            self.cond.notify_all()

    async def disable(self, model):
        """Take a model out of the pool for good (e.g. it does not exist for this account)."""
        async with self.cond:
            if len([m for m in self.order if m != model]) == 0:
                return False
            self.order = [m for m in self.order if m != model]
            self.free[model] = -10**6
            self.cond.notify_all()
            return True


# Zhipu error codes meaning "this model cannot be used by this account": model does not exist,
# no permission, or no balance for a paid model. Such a model is dropped from the pool.
MODEL_UNUSABLE_CODES = ("1211", "1212", "1213", "1220", "1221", "1222", "1113")


def model_unusable(r) -> bool:
    if r is None or r.status_code not in (400, 401, 403, 404, 429):
        return False
    text = r.text[:600]
    return any(f'"{c}"' in text or f"'{c}'" in text or f"code: {c}" in text for c in MODEL_UNUSABLE_CODES)


pool = ModelPool(LLM_POOL)
embed_slots = asyncio.Semaphore(MAX_EMBED)
def upstream_proxy():
    """Use the same proxy as the rest of Windows (e.g. a VPN's system proxy) for the internet API.

    config.py sets NO_PROXY for local addresses; that would also stop httpx from reading the
    Windows proxy settings, so the proxy is looked up here explicitly.
    """
    import urllib.request
    from urllib.parse import urlparse
    if urlparse(LLM_URL).hostname in ("127.0.0.1", "localhost"):
        return None  # a local test server needs no proxy
    env = {k.lower(): v for k, v in os.environ.items()}
    url = env.get("https_proxy") or env.get("all_proxy") or env.get("http_proxy")
    if not url and os.name == "nt":
        try:
            reg = urllib.request.getproxies_registry()
            url = reg.get("https") or reg.get("http")
        except Exception:
            url = None
    if url and "://" not in url:
        url = "http://" + url
    return url or None


UPSTREAM_PROXY = upstream_proxy()
# A healthy answer of up to 4096 tokens arrives in about 1-2 minutes. If nothing has come back
# after LLM_TIMEOUT seconds the connection has stalled (seen with VPNs / a busy free tier), so
# it is dropped and the request is sent again instead of waiting up to 10 minutes.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "150"))
http = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=20), proxy=UPSTREAM_PROXY, trust_env=False)
local_http = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10), trust_env=False)  # Ollama: never via proxy


def log(record: dict) -> None:
    record["ts"] = time.time()
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def thinking_off_body(model: str) -> dict:
    """Fields that switch off 'thinking' mode; only newer GLM models have that mode."""
    if not DISABLE_THINKING:
        return {}
    if "bigmodel" in LLM_URL and not os.getenv("LLM_EXTRA_BODY"):
        return EXTRA_BODY if model.startswith(("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")) else {}
    return EXTRA_BODY


def for_model(body: dict, model: str) -> dict:
    """Copy of a prepared request for one model of the pool."""
    b = dict(body, model=model)
    b.pop("thinking", None)
    b.update(thinking_off_body(model))
    if model.startswith("glm-4-flash"):
        b["max_tokens"] = min(b["max_tokens"], 4000)   # GLM-4-Flash allows at most 4K output tokens
    return b


def prepare_chat(body: dict) -> dict:
    body = {k: v for k, v in body.items() if k in ALLOWED}
    body["model"] = LLM_MODEL
    msgs = [dict(m) for m in body.get("messages", [])]

    # json_schema -> json_object (+ schema written into the prompt)
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        schema = json.dumps(rf.get("json_schema", {}).get("schema", {}), ensure_ascii=False)
        msgs.insert(0, {"role": "system", "content":
                        "Reply with a single valid JSON object that follows this JSON schema "
                        f"exactly (no extra text):\n{schema}"})
        body["response_format"] = {"type": "json_object"}
    elif isinstance(rf, dict) and rf.get("type") not in ("text", "json_object"):
        body.pop("response_format")

    # Zhipu needs at least one user message
    if msgs and not any(m.get("role") == "user" for m in msgs):
        msgs[-1]["role"] = "user"
    body["messages"] = msgs

    # Zhipu accepts temperature 0.0-1.0 and top_p 0.01-1.0
    if "temperature" in body and body["temperature"] is not None:
        body["temperature"] = min(max(float(body["temperature"]), 0.0), 1.0)
    if "top_p" in body and body["top_p"] is not None:
        body["top_p"] = min(max(float(body["top_p"]), 0.01), 1.0)
    body["max_tokens"] = min(int(body.get("max_tokens") or MAX_OUTPUT), MAX_OUTPUT)
    return body


async def post_with_retry(url, payload, headers, attempts=12, client=None):
    """POST with exponential back-off on rate limits (HTTP 429) and temporary server errors."""
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            r = await (client or http).post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            log({"tag": "gateway", "kind": "retry", "status": 0, "attempt": attempt,
                 "error": f"{type(e).__name__}: {e}"[:200], "seconds": 0})
            if attempt == attempts:
                raise
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 60)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
            wait = float(r.headers.get("retry-after", 0) or 0) or delay
            await asyncio.sleep(wait + random.random())
            delay = min(delay * 2, 90)
            continue
        return r, attempt
    return r, attempts


@app.get("/health")
async def health():
    return {"ok": True, "llm": LLM_MODEL, "pool": LLM_POOL, "embed": EMBED_MODEL, "proxy": UPSTREAM_PROXY}


@app.get("/{tag}/v1/models")
async def models(tag: str):
    return {"object": "list", "data": [{"id": LLM_MODEL, "object": "model"},
                                       {"id": EMBED_MODEL, "object": "model"}]}


# GraphRAG asks the LLM to merge the descriptions of every entity/relation found more than once.
# LightRAG only uses the LLM for this when an item has at least 8 description fragments
# (its default FORCE_LLM_SUMMARY_ON_MERGE) and otherwise joins them. Applying the same rule to
# GraphRAG keeps the two graph systems comparable and saves thousands of slow free-tier calls.
MERGE_LLM_MIN = int(os.getenv("GRAPH_MERGE_LLM_MIN_DESCRIPTIONS", "8"))


def local_merge(body: dict):
    """Return the merged description for a small GraphRAG summarisation request, else None."""
    text = "\n".join(m.get("content", "") for m in body.get("messages", []) if isinstance(m.get("content"), str))
    if "Description List:" not in text or "comprehensive summary" not in text:
        return None
    try:
        raw = text.split("Description List:", 1)[1].split("#######", 1)[0].strip()
        descriptions = json.loads(raw)
    except (ValueError, IndexError):
        return None
    if not isinstance(descriptions, list) or len(descriptions) >= MERGE_LLM_MIN:
        return None
    unique = []
    for d in descriptions:
        d = str(d).strip()
        if d and d not in unique:
            unique.append(d)
    words = " ".join(unique).split()
    return " ".join(words[:500])


def completion_response(content: str) -> dict:
    return {"id": "gateway-local", "object": "chat.completion", "created": int(time.time()), "model": LLM_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


@app.post("/{tag}/v1/chat/completions")
async def chat_completions(tag: str, request: Request):
    raw_body = await request.json()
    merged = local_merge(raw_body) if tag.startswith("graphrag") else None
    if merged is not None:
        log({"tag": tag, "kind": "local_merge", "status": 200, "seconds": 0, "prompt_tokens": 0,
             "completion_tokens": 0})
        data = completion_response(merged)
        if raw_body.get("stream"):
            return StreamingResponse(as_sse(data), media_type="text/event-stream")
        return JSONResponse(data)
    body = prepare_chat(raw_body)
    wants_stream = bool(body.pop("stream", False))  # always call upstream without streaming:
    headers = {"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"}
    url = f"{LLM_URL}/chat/completions"             # this gives retries + exact token counts

    t0 = time.time()
    delay, model, r = 5.0, None, None
    for attempts in range(1, 13):
        model = await pool.acquire(avoid=model)
        try:
            r = await http.post(url, json=for_model(body, model), headers=headers)
        except httpx.HTTPError as e:
            r = None
            log({"tag": "gateway", "kind": "retry", "status": 0, "attempt": attempts, "model": model,
                 "error": f"{type(e).__name__}: {e}"[:200], "seconds": 0})
        finally:
            await pool.release(model)
        if model_unusable(r) and await pool.disable(model):
            log({"tag": "gateway", "kind": "model_disabled", "status": r.status_code, "model": model,
                 "error": r.text[:300], "seconds": 0})
            continue
        if r is not None and (r.status_code not in (429, 500, 502, 503, 504) or attempts == 12):
            break
        await asyncio.sleep(delay + random.random())   # rate limit / server busy: wait, then retry
        delay = min(delay * 2, 60)
    seconds = time.time() - t0
    if r is None:
        log({"tag": tag, "kind": "chat", "status": 599, "error": "no response after retries",
             "seconds": round(seconds, 3), "model": model})
        return JSONResponse({"error": {"message": "gateway: upstream not reachable"}}, status_code=502)
    try:
        data = r.json()
    except ValueError:
        data = {"error": {"message": r.text[:500]}}
    usage = data.get("usage") or {}
    log({"tag": tag, "kind": "chat", "status": r.status_code, "attempts": attempts, "model": model,
         "seconds": round(seconds, 3), "stream": wants_stream,
         "prompt_tokens": usage.get("prompt_tokens", 0),
         "completion_tokens": usage.get("completion_tokens", 0),
         "error": None if r.status_code == 200 else str(data)[:300]})
    if r.status_code != 200 or not wants_stream:
        return JSONResponse(data, status_code=r.status_code)
    return StreamingResponse(as_sse(data), media_type="text/event-stream")


def as_sse(data):
    """Send a finished answer to a client that asked for streaming (GraphRAG queries do)."""
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    base = {"id": data.get("id", "gw"), "object": "chat.completion.chunk",
            "created": int(time.time()), "model": data.get("model", LLM_MODEL)}
    first = dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": msg.get("content") or ""},
                                 "finish_reason": None}])
    last = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}], usage=data.get("usage"))
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/{tag}/v1/embeddings")
async def embeddings(tag: str, request: Request):
    body = await request.json()
    texts = body.get("input", [])
    if isinstance(texts, str):
        texts = [texts]
    payload = {"model": EMBED_MODEL, "input": texts}  # Ollama ignores "dimensions"
    async with embed_slots:
        t0 = time.time()
        try:
            r, attempts = await post_with_retry(f"{EMBED_URL}/embeddings", payload, {}, attempts=4,
                                               client=local_http)
        except httpx.HTTPError as e:
            log({"tag": tag, "kind": "embed", "status": 599, "error": f"Ollama not reachable: {e}"[:300],
                 "seconds": round(time.time() - t0, 3)})
            return JSONResponse({"error": {"message": "gateway: Ollama is not reachable at "
                                           f"{EMBED_URL} - is the Ollama app running? ({e})"}}, status_code=502)
        seconds = time.time() - t0
    data = r.json() if r.status_code == 200 else {"error": {"message": r.text[:500]}}
    usage = data.get("usage") or {}
    log({"tag": tag, "kind": "embed", "status": r.status_code, "attempts": attempts,
         "seconds": round(seconds, 3), "inputs": len(texts),
         "characters": sum(len(t) for t in texts),
         "prompt_tokens": usage.get("prompt_tokens", 0)})
    return JSONResponse(data, status_code=r.status_code)


if __name__ == "__main__":
    if not LLM_KEY:
        print("WARNING: LLM_API_KEY is empty - put your Zhipu key in the .env file")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
