"""Step 5 - Evaluate and compare the engines on the same 30 questions.

For every (question, engine) pair it records: the answer, the time taken, and (from the
gateway log, afterwards) how many LLM calls and tokens the answer needed.  An LLM judge
then scores each answer 1-5 against the reference answer.

The four engines are tested at the same time (one worker per engine, each asking its 30
questions one after another), so the per-question LLM calls of an engine can still be told
apart in the gateway log by their tag and time window. Each engine writes its own file
eval/results_<engine>.csv; at the end they are merged into eval/results.csv.
The script can be stopped and started again - finished rows are skipped.
Usage:  python 05_evaluate.py                 (all engines)
        python 05_evaluate.py "Plain RAG"     (only one engine)
"""
import asyncio
import csv
import re
import sys
import time

import pandas as pd

from config import ROOT, chat
from engines import ENGINES, rewrite_question

QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "eval" / "results.csv"
DIALOGUE = ROOT / "eval" / "dialogue_test.csv"
# LightRAG (mix) stays available in the chat app but is not evaluated, to save free-API calls.
TEST_ENGINES = ["Plain RAG", "LightRAG (hybrid)", "GraphRAG (local)", "GraphRAG (global)"]
FIELDS = ["id", "type", "engine", "question", "start_ts", "end_ts", "seconds", "score", "judge_raw", "answer", "error"]

JUDGE = """You are grading an answer produced by a question-answering system about aviation documents.
Compare the ANSWER with the REFERENCE answer and give one integer score:
5 = correct and complete; 4 = correct with minor omissions; 3 = partly correct;
2 = mostly wrong or very incomplete; 1 = wrong, irrelevant, or says it cannot find an answer that exists.
If the REFERENCE says the information is not in the documents, give 5 when the ANSWER says it cannot find it,
and 1 when the ANSWER invents an answer.
Reply with the score only.

QUESTION: {q}
REFERENCE: {ref}
ANSWER: {ans}"""

DIALOGUES = [  # multi-turn conversations to test the dialogue (follow-up) handling
    ["What is the maximum design takeoff weight of the Boeing 787-8?", "And the 787-9?",
     "Which of the two can carry more passengers in dual class?"],
    ["Which engine does the Boeing 737 MAX use?", "What about the A320neo?"],
]


def judge(question, reference, answer):
    raw = chat([{"role": "user", "content": JUDGE.format(q=question, ref=reference, ans=answer[:4000])}],
               tag="judge", temperature=0.0, max_tokens=10)
    m = re.search(r"[1-5]", raw)
    return (int(m.group()) if m else None), raw


def engine_file(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return ROOT / "eval" / f"results_{slug}.csv"


async def run_engine(name):
    qs = pd.read_csv(QUESTIONS)
    out = engine_file(name)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = {(int(r.id), r.engine) for r in old.itertuples() if pd.isna(r.error) or r.error == ""}
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        if True:
            for q in qs.itertuples():
                if (int(q.id), name) in done:
                    continue
                row = {"id": q.id, "type": q.type, "engine": name, "question": q.question,
                       "start_ts": time.time(), "answer": "", "error": ""}
                try:
                    res = await ENGINES[name](q.question)
                    row["answer"] = res["answer"]
                except Exception as e:
                    row["error"] = str(e)[:500]
                row["end_ts"] = time.time()
                row["seconds"] = round(row["end_ts"] - row["start_ts"], 2)
                if not row["error"]:
                    row["score"], row["judge_raw"] = await asyncio.to_thread(
                        judge, q.question, q.reference, row["answer"])
                w.writerow(row)
                f.flush()
                print(f"[{name}] Q{q.id} ({q.type}) score={row.get('score')} {row['seconds']}s {row['error'][:80]}")


async def ensure_lightrag():
    """Rebuild the LightRAG index first if its last build left failed documents."""
    from lightrag_setup import lightrag_status
    for attempt in range(3):
        done, total = lightrag_status()
        if total and done == total:
            return
        print(f"LightRAG index incomplete ({done}/{total} documents) - building it before its questions")
        proc = await asyncio.create_subprocess_exec(sys.executable, "04_build_lightrag.py", cwd=ROOT)
        await proc.wait()
    raise RuntimeError("LightRAG index could not be built")


async def run_engine_when_ready(name):
    if name.startswith("LightRAG"):
        await ensure_lightrag()
    await run_engine(name)


async def run_questions(engines):
    await asyncio.gather(*(run_engine_when_ready(n) for n in engines))
    parts = [pd.read_csv(engine_file(n)) for n in TEST_ENGINES if engine_file(n).exists()]
    if parts:  # keep the last (successful) row of every (question, engine) pair
        allr = pd.concat(parts, ignore_index=True)
        allr["ok"] = allr.error.isna() | (allr.error == "")
        allr = allr.sort_values(["ok", "end_ts"]).drop_duplicates(["id", "engine"], keep="last")
        allr.drop(columns="ok").sort_values(["engine", "id"]).to_csv(RESULTS, index=False, encoding="utf-8")


async def run_dialogues(engine="LightRAG (hybrid)"):
    rows = []
    for d, turns in enumerate(DIALOGUES, 1):
        history = []
        for t, msg in enumerate(turns, 1):
            standalone = await asyncio.to_thread(rewrite_question, msg, history)
            res = await ENGINES[engine](standalone)
            history += [{"role": "user", "content": msg}, {"role": "assistant", "content": res["answer"]}]
            rows.append({"dialogue": d, "turn": t, "user_message": msg, "rewritten_question": standalone,
                         "engine": engine, "answer": res["answer"]})
            print(f"Dialogue {d}.{t}: {msg!r} -> {standalone!r}")
    pd.DataFrame(rows).to_csv(DIALOGUE, index=False, encoding="utf-8")


async def main():
    engines = sys.argv[1:] or TEST_ENGINES
    await run_questions(engines)
    if not sys.argv[1:]:
        await run_dialogues()
    print("\nDone -> eval/results.csv and eval/dialogue_test.csv")


if __name__ == "__main__":
    asyncio.run(main())
