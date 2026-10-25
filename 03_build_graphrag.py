"""Step 3 - Build the GraphRAG index with Microsoft GraphRAG (the library on graphrag.homes).

What GraphRAG does during indexing (all automatic, run by `graphrag index`):
  1. split the text into 1200-token chunks
  2. LLM extracts entities + relationships from every chunk
  3. merge duplicates and summarise each entity's descriptions (LLM)
  4. Leiden community detection on the graph
  5. LLM writes a report for every community
  6. embed entity descriptions and chunks (bge-m3)

Usage:  python 03_build_graphrag.py            (build / resume)
        python 03_build_graphrag.py --tune     (also auto-tune the prompts to aviation first)
"""
import json
import shutil
import subprocess
import sys
import time

import pandas as pd

from config import DATA_TXT, GRAPHRAG_DIR, METRICS_DIR, ROOT, ensure_gateway, folder_size_mb, graphrag_env

TAG = "graphrag_index"


def run(args, env):
    print(">", " ".join(args))
    subprocess.run([sys.executable, "-m", "graphrag", *args], env=env, cwd=ROOT, check=True)


def main():
    ensure_gateway()
    env = graphrag_env(TAG)

    # 1. create the project once (prompts/, settings.yaml), then use our settings
    if not (GRAPHRAG_DIR / "prompts").exists():
        run(["init", "--root", str(GRAPHRAG_DIR), "--model", env["GRAPHRAG_LLM_MODEL"],
             "--embedding", env["GRAPHRAG_EMBED_MODEL"], "--force"], env)
    shutil.copy(ROOT / "configs" / "graphrag_settings.yaml", GRAPHRAG_DIR / "settings.yaml")

    # 2. the same text files the other systems use
    (GRAPHRAG_DIR / "input").mkdir(exist_ok=True)
    for f in DATA_TXT.glob("*.txt"):
        shutil.copy(f, GRAPHRAG_DIR / "input" / f.name)

    # 3. optional: adapt the extraction prompts to the aviation domain
    if "--tune" in sys.argv:
        run(["prompt-tune", "--root", str(GRAPHRAG_DIR), "--domain",
             "commercial aircraft manuals and aviation safety", "--limit", "15"], env)

    # 4. build the index (re-running resumes from the LLM cache)
    t0 = time.time()
    run(["index", "--root", str(GRAPHRAG_DIR)], env)
    seconds = time.time() - t0

    # 5. record what was built
    out = GRAPHRAG_DIR / "output"
    counts = {}
    for name in ["documents", "text_units", "entities", "relationships", "communities", "community_reports"]:
        p = out / f"{name}.parquet"
        counts[name] = int(len(pd.read_parquet(p))) if p.exists() else 0
    stats = {"system": "GraphRAG", "seconds": round(seconds, 1), **counts,
             "index_size_mb": folder_size_mb(out)}
    stats_file = out / "stats.json"
    if stats_file.exists():
        stats["pipeline"] = {k: v.get("overall") for k, v in
                             json.loads(stats_file.read_text(encoding="utf-8")).get("workflows", {}).items()}
    (METRICS_DIR / "graphrag_index.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nGraphRAG index built in {seconds/60:.1f} min: {counts}")


if __name__ == "__main__":
    main()
