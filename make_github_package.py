"""Collects the files for the public GitHub repository into github_upload/.

    github_upload/repo/                  -> upload these files to the repository
    github_upload/prebuilt_indexes.zip   -> attach this file to a GitHub release

The API key (.env), the virtual environment, the PDFs and the caches are never copied.
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "github_upload"
REPO = OUT / "repo"

FILES = ["README.md", ".gitignore", ".env.example", "requirements.txt", "corpus.json",
         "setup.ps1", "run_all.ps1", "config.py", "gateway.py", "engines.py", "app.py",
         "lightrag_setup.py", "show_graph.py", "check_setup.py", "make_github_package.py",
         "01_prepare_data.py", "02_build_rag.py", "03_build_graphrag.py", "04_build_lightrag.py",
         "05_evaluate.py", "06_make_report_assets.py",
         "configs/graphrag_settings.yaml", "configs/llm_pool.txt",
         "eval/questions.csv", "eval/results.csv", "eval/dialogue_test.csv",
         "eval/results_plain_rag.csv", "eval/results_graphrag_local.csv",
         "eval/results_graphrag_global.csv", "eval/results_lightrag_hybrid.csv",
         "metrics/corpus_stats.csv", "metrics/rag_index.json", "metrics/graphrag_index.json",
         "metrics/lightrag_index.json", "metrics/summary.json",
         "logs/gateway_calls.jsonl", "report/HW2_2024380178.pdf"]
FOLDERS = ["data/txt", "tiktoken_cache", "report/figures"]

# what the chatbot needs at question time
INDEX_FOLDERS = ["rag_store", "lightrag_store", "graphrag_proj/output", "graphrag_proj/prompts"]
INDEX_FILES = ["graphrag_proj/settings.yaml"]
SKIP_IN_INDEX = ("_failed_build_", "kv_store_llm_response_cache")
FORBIDDEN = (".env",)  # never published


def copy(rel):
    src, dst = ROOT / rel, REPO / rel
    if not src.exists():
        print("  (missing, skipped)", rel)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    REPO.mkdir(parents=True)
    for rel in FILES:
        copy(rel)
    for folder in FOLDERS:
        for f in (ROOT / folder).rglob("*"):
            if f.is_file():
                copy(f.relative_to(ROOT).as_posix())
    for f in REPO.rglob("*"):
        if f.name in FORBIDDEN:
            raise SystemExit(f"refusing to publish {f}")

    with zipfile.ZipFile(OUT / "prebuilt_indexes.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for rel in INDEX_FILES:
            z.write(ROOT / rel, rel)
        for folder in INDEX_FOLDERS:
            for f in (ROOT / folder).rglob("*"):
                rel = f.relative_to(ROOT).as_posix()
                if f.is_file() and not any(s in rel for s in SKIP_IN_INDEX) and f.name not in FORBIDDEN:
                    z.write(f, rel)

    n = sum(1 for f in REPO.rglob("*") if f.is_file())
    size = (OUT / "prebuilt_indexes.zip").stat().st_size / 1e6
    print(f"\n{n} files in {REPO}")
    print(f"prebuilt_indexes.zip: {size:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
