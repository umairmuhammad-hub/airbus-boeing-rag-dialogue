"""The three knowledge-augmentation engines behind one interface.

    answer = await ENGINES["Plain RAG"](question)   ->  {"answer": ..., "sources": [...]}

Engines:
  Plain RAG          - top-5 similar chunks (FAISS) + GLM answer
  GraphRAG (local)   - Microsoft GraphRAG local search: entities near the question + neighbours
  GraphRAG (global)  - Microsoft GraphRAG global search: map-reduce over community reports
  LightRAG (hybrid)  - HKUDS LightRAG dual-level retrieval (entities + relations)
  LightRAG (mix)     - LightRAG graph retrieval + vector chunks (its default mode)
"""
import asyncio
import json
import os
import sys

import faiss
import numpy as np
from lightrag import QueryParam

from config import GRAPHRAG_DIR, RAG_DIR, ROOT, TOP_K, chat, embed, ensure_gateway, graphrag_env

SYSTEM_PROMPT = (
    "You are an aviation documentation assistant for Airbus and Boeing documents. "
    "Answer ONLY with information from the context. If the context does not contain "
    "the answer, say that you could not find it in the documents. Give numbers with "
    "their units and mention which document the information comes from.")


# ---------------- 1. Plain RAG ----------------
class PlainRAG:
    def __init__(self):
        self.index = faiss.read_index(str(RAG_DIR / "index.faiss"))
        self.chunks = json.loads((RAG_DIR / "chunks.json").read_text(encoding="utf-8"))

    def retrieve(self, question, k=TOP_K):
        q = np.asarray(embed([question], tag="rag_query"), dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        return [dict(self.chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0])]

    async def __call__(self, question):
        hits = await asyncio.to_thread(self.retrieve, question)
        context = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
        answer = await asyncio.to_thread(chat, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}], "rag_query")
        return {"answer": answer, "sources": sorted({h["source"] for h in hits})}


# ---------------- 2. GraphRAG (official CLI) ----------------
# Community level used by GraphRAG search (0 = a few large communities ... higher = many small ones).
# Level 1 keeps global search to a few map calls per question, which matters on a free API
# that serves one request at a time.
COMMUNITY_LEVEL = os.getenv("GRAPHRAG_COMMUNITY_LEVEL", "1")


async def graphrag_answer(question, method="local"):
    ensure_gateway()  # the GraphRAG command talks to the gateway of this process
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "graphrag", "query", question,
        "--root", str(GRAPHRAG_DIR), "--method", method, "--community-level", COMMUNITY_LEVEL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=graphrag_env(f"graphrag_query_{method}"), cwd=ROOT)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("GraphRAG query failed: " + err.decode("utf-8", "ignore")[-800:])
    return {"answer": out.decode("utf-8", "ignore").strip(),
            "sources": [f"GraphRAG {method} search (entities, relations, community reports)"]}


# ---------------- 3. LightRAG ----------------
_lightrag = None
_lightrag_lock = asyncio.Lock()


async def lightrag_answer(question, mode="hybrid"):
    global _lightrag
    async with _lightrag_lock:
        if _lightrag is None:
            from lightrag_setup import make_rag
            _lightrag = await make_rag("lightrag_query")
    answer = await _lightrag.aquery(question, param=QueryParam(mode=mode, enable_rerank=False))
    return {"answer": str(answer).strip(), "sources": [f"LightRAG {mode} mode (entities, relations, chunks)"]}


_plain = None


async def plain_rag_answer(question):
    global _plain
    if _plain is None:
        _plain = PlainRAG()
    return await _plain(question)


ENGINES = {
    "Plain RAG": plain_rag_answer,
    "GraphRAG (local)": lambda q: graphrag_answer(q, "local"),
    "GraphRAG (global)": lambda q: graphrag_answer(q, "global"),
    "LightRAG (hybrid)": lambda q: lightrag_answer(q, "hybrid"),
    "LightRAG (mix)": lambda q: lightrag_answer(q, "mix"),
}


# ---------------- dialogue: follow-up question -> standalone question ----------------
def rewrite_question(question, history):
    """Use the chat history to turn a follow-up ("and the 787-9?") into a full question."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history[-6:])
    rewritten = chat([{"role": "user", "content":
                       "Conversation so far:\n" + convo +
                       "\n\nRewrite the user's last question so it can be understood without the "
                       "conversation (resolve words like 'it', 'that one', 'its'). Return only the "
                       f"rewritten question.\n\nLast question: {question}"}],
                     tag="dialogue", temperature=0.0, max_tokens=200)
    return rewritten.strip().strip('"') or question
