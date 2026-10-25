"""Step 4 - Build the LightRAG index (HKUDS/LightRAG).

What LightRAG does during indexing:
  1. split the text into 1200-token chunks
  2. LLM extracts entities + relationships from every chunk, each with keywords
  3. merge duplicate entities/relations (LLM summary when descriptions are long)
  4. embed entities, relations and chunks into three vector stores
No community detection and no community reports -> cheaper than GraphRAG.
"""
import asyncio
import json
import shutil
import sys
import time

import networkx as nx

from config import DATA_TXT, LIGHTRAG_DIR, METRICS_DIR, folder_size_mb
from lightrag_setup import lightrag_status, make_rag

KEEP = "kv_store_llm_response_cache.json"   # cached extraction answers: reused, never recomputed


def clear_failed_build():
    """A build in which documents failed is moved aside; only the extraction cache is kept."""
    done, total = lightrag_status()
    if total == 0 or done == total:
        return
    backup = LIGHTRAG_DIR / f"_failed_build_{int(time.time())}"
    backup.mkdir()
    for f in LIGHTRAG_DIR.iterdir():
        if f.is_file() and f.name != KEEP:
            shutil.move(str(f), backup / f.name)
    print(f"Previous LightRAG build had {total - done} failed documents - rebuilding (cache kept)")


async def main():
    clear_failed_build()
    rag = await make_rag("lightrag_index")
    files = sorted(DATA_TXT.glob("*.txt"))
    t0 = time.time()
    try:
        await rag.ainsert([f.read_text(encoding="utf-8") for f in files],
                          ids=[f.stem for f in files], file_paths=[f.name for f in files])
    finally:
        await rag.finalize_storages()
    seconds = time.time() - t0
    done, total = lightrag_status()
    if done < len(files):
        print(f"LightRAG: only {done} of {len(files)} documents were processed - see the errors above")
        sys.exit(1)

    graph_file = LIGHTRAG_DIR / "graph_chunk_entity_relation.graphml"
    G = nx.read_graphml(graph_file) if graph_file.exists() else nx.Graph()
    chunks_file = LIGHTRAG_DIR / "kv_store_text_chunks.json"
    n_chunks = len(json.loads(chunks_file.read_text(encoding="utf-8"))) if chunks_file.exists() else 0
    stats = {"system": "LightRAG", "seconds": round(seconds, 1), "documents": len(files),
             "text_units": n_chunks, "entities": G.number_of_nodes(),
             "relationships": G.number_of_edges(), "index_size_mb": folder_size_mb(LIGHTRAG_DIR)}
    (METRICS_DIR / "lightrag_index.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nLightRAG index built in {seconds/60:.1f} min: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
