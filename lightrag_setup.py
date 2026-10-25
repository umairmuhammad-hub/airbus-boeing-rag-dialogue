"""Creates the LightRAG object (HKUDS LightRAG, the teacher's GitHub link).

Based on the repository's own example `examples/lightrag_openai_compatible_demo.py`,
changed so that both the LLM and the embeddings go through our local gateway.
"""
from functools import partial

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from config import (CHUNK_OVERLAP, CHUNK_TOKENS, EMBED_DIM, EMBED_MODEL, LIGHTRAG_DIR, LLM_MODEL,
                    base_url, ensure_gateway)


def make_llm_func(tag):
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
        kwargs.pop("hashing_kv", None)
        return await openai_complete_if_cache(
            LLM_MODEL, prompt, system_prompt=system_prompt, history_messages=history_messages,
            api_key="via-gateway", base_url=base_url(tag), **kwargs)
    return llm_model_func


async def make_rag(tag: str) -> LightRAG:
    """tag = 'lightrag_index' while building, 'lightrag_query' while answering."""
    ensure_gateway()
    LIGHTRAG_DIR.mkdir(exist_ok=True)
    rag = LightRAG(
        working_dir=str(LIGHTRAG_DIR),
        llm_model_func=make_llm_func(tag),
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM, max_token_size=8192,
            func=partial(openai_embed.func, model=EMBED_MODEL,
                         base_url=base_url(tag), api_key="via-gateway")),
        chunk_token_size=CHUNK_TOKENS,               # same chunking as RAG and GraphRAG
        chunk_overlap_token_size=CHUNK_OVERLAP,
        tiktoken_model_name="gpt-4",                 # = cl100k_base tokenizer, same as the others
        entity_extract_max_gleaning=0,               # one extraction pass, same as GraphRAG setting
        llm_model_max_async=6,                       # parallel requests (gateway limits per model)
        embedding_func_max_async=1,                  # Ollama on a CPU: one embedding batch at a time
        default_embedding_timeout=900,               # CPU embeddings of a full batch can take minutes
        default_llm_timeout=600,                     # free-tier calls may wait in a queue
        enable_llm_cache=False,                      # no cached answers -> honest query timings
        enable_llm_cache_for_entity_extract=True,    # lets an interrupted build resume cheaply
    )
    await rag.initialize_storages()
    return rag


def lightrag_status():
    """(processed documents, all documents) according to LightRAG's document status store."""
    import json
    f = LIGHTRAG_DIR / "kv_store_doc_status.json"
    if not f.exists():
        return 0, 0
    st = json.loads(f.read_text(encoding="utf-8"))
    return sum(1 for v in st.values() if str(v.get("status", "")).lower() == "processed"), len(st)
