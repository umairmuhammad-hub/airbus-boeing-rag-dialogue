"""Step 2 - Build the plain RAG index (baseline).

Indexing pipeline:  text -> 1200-token chunks (100 overlap) -> bge-m3 vectors -> FAISS index.
No LLM call is needed to build it, only embeddings.
"""
import json
import time

import faiss
import numpy as np
import tiktoken

from config import CHUNK_OVERLAP, CHUNK_TOKENS, DATA_TXT, METRICS_DIR, RAG_DIR, TOKENIZER, embed

TAG = "rag_index"  # gateway tag: every embedding call is logged under this name


def chunk_tokens(text, size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """Split text into windows of `size` tokens that overlap by `overlap` tokens
    (the same scheme GraphRAG and LightRAG use)."""
    enc = tiktoken.get_encoding(TOKENIZER)
    ids = enc.encode(text)
    step = size - overlap
    return [enc.decode(ids[i:i + size]) for i in range(0, max(len(ids) - overlap, 1), step)]


def main():
    t0 = time.time()
    chunks = []
    for f in sorted(DATA_TXT.glob("*.txt")):
        for i, piece in enumerate(chunk_tokens(f.read_text(encoding="utf-8"))):
            chunks.append({"source": f.stem, "chunk": i, "text": piece})
    print(f"{len(chunks)} chunks from {len(list(DATA_TXT.glob('*.txt')))} documents")

    vectors = []
    for start in range(0, len(chunks), 16):  # embed 16 chunks per request
        vectors += embed([c["text"] for c in chunks[start:start + 16]], tag=TAG)
        print(f"  embedded {min(start + 16, len(chunks))}/{len(chunks)}", end="\r")
    X = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(X)                    # unit length -> inner product = cosine similarity
    index = faiss.IndexFlatIP(X.shape[1])    # exact search (fast enough for a few hundred chunks)
    index.add(X)

    RAG_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(RAG_DIR / "index.faiss"))
    (RAG_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    stats = {"system": "RAG", "seconds": round(time.time() - t0, 1), "chunks": len(chunks),
             "vector_dim": int(X.shape[1])}
    (METRICS_DIR / "rag_index.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nRAG index built in {stats['seconds']} s -> rag_store/")


if __name__ == "__main__":
    main()
