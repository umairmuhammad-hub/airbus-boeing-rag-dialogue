"""Step 0 - Check that the free LLM (Zhipu GLM) and the embedding model (Ollama bge-m3) work."""
import sys

from config import LLM_MODEL, EMBED_MODEL, chat, embed


def main():
    try:
        v = embed(["test sentence"], tag="setup_check")
        print(f"OK  embeddings: {EMBED_MODEL} returns vectors of length {len(v[0])}")
    except Exception as e:
        print(f"FAILED embeddings ({EMBED_MODEL}): {e}\n -> Is Ollama running? Did 'ollama pull bge-m3' finish?")
        sys.exit(1)
    try:
        import time
        t0 = time.time()
        a = chat([{"role": "user", "content": "Reply with exactly: OK"}], tag="setup_check", max_tokens=10)
        print(f"OK  LLM: {LLM_MODEL} replied {a!r} in {time.time() - t0:.1f} s")
    except Exception as e:
        print(f"FAILED LLM ({LLM_MODEL}): {e}\n -> Check LLM_API_KEY in .env (no spaces, no quotes).")
        sys.exit(1)


if __name__ == "__main__":
    main()
