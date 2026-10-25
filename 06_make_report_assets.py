"""Step 6 - Turn the measurements into the report's tables, figures and numbers.

Reads:   metrics/*.json, metrics/corpus_stats.csv, logs/gateway_calls.jsonl, eval/results.csv
Writes:  report/generated/*.tex   (LaTeX tables + \\newcommand numbers used in the text)
         report/figures/*.pdf     (charts and knowledge-graph pictures)
         metrics/summary.json     (everything in one file)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from config import GRAPHRAG_DIR, LIGHTRAG_DIR, LOG_DIR, METRICS_DIR, ROOT

GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ENGINES = ["Plain RAG", "GraphRAG (local)", "GraphRAG (global)", "LightRAG (hybrid)", "LightRAG (mix)"]
COLORS = dict(zip(ENGINES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))  # fixed order
SYSTEMS = {"RAG": "rag", "GraphRAG": "graphrag", "LightRAG": "lightrag"}
SYS_COLORS = {"RAG": "#2a78d6", "GraphRAG": "#eb6834", "LightRAG": "#eda100"}
QUERY_TAG = {"Plain RAG": "rag_query", "GraphRAG (local)": "graphrag_query_local",
             "GraphRAG (global)": "graphrag_query_global",
             "LightRAG (hybrid)": "lightrag_query", "LightRAG (mix)": "lightrag_query"}
TYPES = ["factual", "multi-hop", "global", "unanswerable"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a",
                     "ytick.color": "#3d3d3a", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "pdf.fonttype": 42})


def load_log():
    f = LOG_DIR / "gateway_calls.jsonl"
    rows = [json.loads(l) for l in f.open(encoding="utf-8")] if f.exists() else []
    df = pd.DataFrame(rows)
    for c in ["prompt_tokens", "completion_tokens", "seconds", "inputs"]:
        if c not in df:
            df[c] = 0
    return df.fillna({"prompt_tokens": 0, "completion_tokens": 0, "inputs": 0})


def tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{nd}f}"


def write_table(name, header, rows, colspec, caption, label):
    if not rows:
        rows = [[r"\multicolumn{%d}{c}{\itshape pending final run}" % len(header)]]
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = (f"\\begin{{table}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\\small\n"
           f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n"
           f"\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\end{{table}}\n")
    (GEN / f"{name}.tex").write_text(tex, encoding="utf-8")


def main():
    log = load_log()
    ok = log[log.get("status", 200) == 200] if len(log) else log
    numbers = {}

    # ---------- corpus ----------
    corpus = pd.read_csv(METRICS_DIR / "corpus_stats.csv")
    write_table("corpus_table", ["Document", "Maker", "Pages used", "Words"],
                [[tex_escape(r.title), r.maker, fmt(r.pages_used), fmt(r.words)] for r in corpus.itertuples()]
                + [[r"\textbf{Total}", "", fmt(corpus.pages_used.sum()), fmt(corpus.words.sum())]],
                "p{8.2cm}lrr", "The knowledge base: official Airbus and Boeing documents.", "tab:corpus")
    numbers.update(CorpusDocs=len(corpus), CorpusPages=int(corpus.pages_used.sum()),
                   CorpusWords=f"{int(corpus.words.sum()):,}")

    # ---------- indexing efficiency ----------
    idx_rows, idx = [], {}
    for name, key in SYSTEMS.items():
        f = METRICS_DIR / f"{key}_index.json"
        if not f.exists():  # this index is not finished yet
            idx_rows.append([name, r"\multicolumn{9}{c}{\itshape pending}"])
            continue
        m = json.loads(f.read_text())
        part = ok[ok.tag == f"{key}_index"] if len(ok) else ok
        chat = part[part.kind == "chat"] if len(part) else part
        emb = part[part.kind == "embed"] if len(part) else part
        # The graph indexes were built over several runs (stops, restarts, cached resumption), so the
        # time reported is the total time spent waiting for model calls, summed from the gateway log.
        busy = (chat.seconds.sum() if len(chat) else 0) + (emb.seconds.sum() if len(emb) else 0)
        d = {"minutes": busy / 60 if busy else m.get("seconds", np.nan) / 60, "llm_calls": len(chat),
             "prompt_tokens": int(chat.prompt_tokens.sum()) if len(chat) else 0,
             "completion_tokens": int(chat.completion_tokens.sum()) if len(chat) else 0,
             "embed_requests": len(emb), "embedded_texts": int(emb.inputs.sum()) if len(emb) else 0,
             "chunks": m.get("chunks", m.get("text_units")), "entities": m.get("entities"),
             "relationships": m.get("relationships"), "communities": m.get("community_reports"),
             "size_mb": m.get("index_size_mb")}
        d["llm_tokens"] = d["prompt_tokens"] + d["completion_tokens"]
        idx[name] = d
        idx_rows.append([name, fmt(d["minutes"], 1), fmt(d["llm_calls"]), fmt(d["llm_tokens"]),
                         fmt(d["embedded_texts"]), fmt(d["chunks"]), fmt(d["entities"]),
                         fmt(d["relationships"]), fmt(d["communities"]), fmt(d["size_mb"], 1)])
        numbers[f"{name}IndexMin"] = round(d["minutes"], 1) if not np.isnan(d["minutes"]) else "--"
        numbers[f"{name}IndexCalls"] = f"{d['llm_calls']:,}"
        numbers[f"{name}IndexTokens"] = f"{d['llm_tokens']:,}"
    write_table("index_table",
                ["System", "Model time (min)", "LLM calls", "LLM tokens", "Embedded", "Chunks", "Entities",
                 "Relations", "Comm.", "Size (MB)"], idx_rows, "lrrrrrrrrr",
                "Indexing cost of the three systems on the same corpus (measured by the gateway).", "tab:index")

    # ---------- query efficiency + quality ----------
    res_file = ROOT / "eval" / "results.csv"
    res = pd.read_csv(res_file) if res_file.exists() else pd.DataFrame(
        columns=["id", "type", "engine", "start_ts", "end_ts", "seconds", "score", "error"])
    res = res[res.engine.isin(ENGINES)].drop_duplicates(["id", "engine"], keep="last")
    calls, toks = [], []
    for r in res.itertuples():
        w = ok[(ok.tag == QUERY_TAG[r.engine]) & (ok.ts >= r.start_ts) & (ok.ts <= r.end_ts + 0.05)] if len(ok) else ok
        c = w[w.kind == "chat"] if len(w) else w
        calls.append(len(c))
        toks.append(int(c.prompt_tokens.sum() + c.completion_tokens.sum()) if len(c) else 0)
    res["llm_calls"], res["llm_tokens"] = calls, toks
    good = res[res.error.isna() | (res.error == "")] if len(res) else res

    q_rows, qual_rows, q = [], [], {}
    for e in ENGINES:
        g = good[good.engine == e]
        if g.empty:
            continue
        d = {"latency_mean": g.seconds.mean(), "latency_median": g.seconds.median(),
             "calls": g.llm_calls.mean(), "tokens": g.llm_tokens.mean(), "score": g.score.mean(),
             "good_share": (g.score >= 4).mean() * 100, "errors": int((res.engine == e).sum() - len(g))}
        for t in TYPES:
            d[t] = g[g.type == t].score.mean()
        q[e] = d
        q_rows.append([e, fmt(d["latency_mean"], 1), fmt(d["latency_median"], 1), fmt(d["calls"], 1),
                       fmt(d["tokens"]), fmt(d["errors"])])
        qual_rows.append([e] + [fmt(d[t], 2) for t in TYPES] + [fmt(d["score"], 2), fmt(d["good_share"], 0) + r"\%"])
        tag = e.replace(" ", "").replace("(", "").replace(")", "")
        numbers[f"{tag}Latency"] = round(d["latency_mean"], 1)
        numbers[f"{tag}Tokens"] = f"{d['tokens']:,.0f}"
        numbers[f"{tag}Score"] = round(d["score"], 2)
    write_table("query_table", ["Engine", "Mean (s)", "Median (s)", "LLM calls/q", "Tokens/q", "Errors"],
                q_rows, "lrrrrr", "Query-time efficiency over the 30 test questions.", "tab:query")
    write_table("quality_table", ["Engine", "Factual", "Multi-hop", "Global", "Unans.", "All", r"Score$\geq$4"],
                qual_rows, "lrrrrrr",
                "Answer quality (LLM-judge score 1--5, mean) by question type.", "tab:quality")
    numbers["NumQuestions"] = int(res.id.nunique()) if len(res) else 30

    # ---------- figures ----------
    if len(idx) >= 2:  # a comparison needs at least two finished indexes
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
        names = list(idx)
        for ax, key, title in [(axes[0], "llm_tokens", "LLM tokens to build the index"),
                               (axes[1], "minutes", "Time to build the index (min)")]:
            vals = [idx[n][key] if not (isinstance(idx[n][key], float) and np.isnan(idx[n][key])) else 0 for n in names]
            bars = ax.bar(names, vals, color=[SYS_COLORS[n] for n in names], width=0.55, edgecolor="white", linewidth=2)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:,.0f}" if key == "llm_tokens" else f"{v:,.1f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=8, color="#3d3d3a", xytext=(0, 2), textcoords="offset points")
            ax.set_title(title, fontsize=9, loc="left")
            ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(FIG / "fig_index_cost.pdf")
        plt.close(fig)

    if q:
        fig, ax = plt.subplots(figsize=(6.6, 2.6))
        x = np.arange(len(TYPES))
        es = [e for e in ENGINES if e in q]
        wd = 0.8 / len(es)
        for i, e in enumerate(es):
            ax.bar(x + (i - (len(es) - 1) / 2) * wd, [q[e][t] for t in TYPES], wd, label=e,
                   color=COLORS[e], edgecolor="white", linewidth=1.5)
        ax.set_xticks(x, ["Factual", "Multi-hop", "Global", "Unanswerable"])
        ax.set_ylim(0, 5.3)
        ax.set_ylabel("Mean judge score (1-5)")
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        fig.tight_layout()
        fig.savefig(FIG / "fig_quality_by_type.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 2.6))
        for e in es:
            ax.scatter(q[e]["tokens"], q[e]["score"], s=60, color=COLORS[e], edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(e, (q[e]["tokens"], q[e]["score"]), xytext=(6, 4), textcoords="offset points", fontsize=8,
                        color="#3d3d3a")
        ax.set_xscale("log")
        ax.set_xlabel("LLM tokens per question (log scale)")
        ax.set_ylabel("Mean judge score (1-5)")
        fig.tight_layout()
        fig.savefig(FIG / "fig_cost_vs_quality.pdf")
        plt.close(fig)

    # ---------- knowledge-graph pictures ----------
    for name, path in [("graphrag", GRAPHRAG_DIR / "output" / "graph.graphml"),
                       ("lightrag", LIGHTRAG_DIR / "graph_chunk_entity_relation.graphml")]:
        if not path.exists():
            continue
        G = nx.read_graphml(path).to_undirected()
        top = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:45]]
        H = G.subgraph(top)
        pos = nx.spring_layout(H, seed=7, k=0.9)
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.axis("off")
        deg = dict(H.degree)
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#c3c2b7", width=0.6)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[30 + 18 * deg[n] for n in H],
                               node_color=SYS_COLORS["GraphRAG" if name == "graphrag" else "LightRAG"],
                               edgecolors="white", linewidths=1)
        labels = {n: str(n)[:22] for n in top[:18]}
        nx.draw_networkx_labels(H, pos, labels, ax=ax, font_size=6.5, font_color="#1f1f1d")
        fig.tight_layout()
        fig.savefig(FIG / f"fig_{name}_graph.pdf")
        plt.close(fig)

    # ---------- most connected entities of each graph ----------
    ent_rows = []
    tops = {}
    for name, path in [("GraphRAG", GRAPHRAG_DIR / "output" / "graph.graphml"),
                       ("LightRAG", LIGHTRAG_DIR / "graph_chunk_entity_relation.graphml")]:
        if path.exists():
            G = nx.read_graphml(path).to_undirected()
            tops[name] = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
            numbers[f"{name}GraphNodes"] = f"{G.number_of_nodes():,}"
            numbers[f"{name}GraphEdges"] = f"{G.number_of_edges():,}"
    for i in range(10):
        row = [str(i + 1)]
        for name in ["GraphRAG", "LightRAG"]:
            t = tops.get(name, [])
            row += [tex_escape(str(t[i][0])[:38]), str(t[i][1])] if i < len(t) else ["--", "--"]
        ent_rows.append(row)
    write_table("entities_table", ["\\#", "GraphRAG entity", "Degree", "LightRAG entity", "Degree"],
                ent_rows if tops else [], "rlrlr",
                "The ten most connected entities in each knowledge graph.", "tab:entities")

    # ---------- engineering statistics from the gateway log ----------
    eng = []
    for tag, label in [("rag_index", "RAG indexing"), ("graphrag_index", "GraphRAG indexing"),
                       ("lightrag_index", "LightRAG indexing"), ("rag_query", "RAG queries"),
                       ("graphrag_query_local", "GraphRAG local queries"),
                       ("graphrag_query_global", "GraphRAG global queries"), ("lightrag_query", "LightRAG queries"),
                       ("dialogue", "Dialogue rewriting"), ("judge", "LLM judge")]:
        part = log[log.tag == tag] if len(log) else log
        if part is None or len(part) == 0:
            continue
        chat = part[part.kind == "chat"]
        emb = part[part.kind == "embed"]
        att = chat["attempts"].fillna(1) if "attempts" in chat else pd.Series(dtype=float)
        eng.append([label, fmt(len(chat)), fmt(int((att > 1).sum()) if len(att) else 0),
                    fmt(chat.seconds.median() if len(chat) else np.nan, 1),
                    fmt(chat.completion_tokens.max() if len(chat) else np.nan),
                    fmt(len(emb)), fmt(emb.seconds.median() if len(emb) else np.nan, 1)])
    write_table("engineering_table", ["Phase", "LLM calls", "Retried", "Median s/call", "Max output tok.",
                                      "Embed req.", "Median s/embed"], eng, "lrrrrrr",
                "Behaviour of the free LLM API and the local embedding model, from the gateway log.", "tab:engineering")

    if len(log):
        numbers["GraphRAGLocalMerges"] = f"{int(((log.tag == 'graphrag_index') & (log.kind == 'local_merge')).sum()):,}"

    # ---------- the question set (appendix) ----------
    qs = pd.read_csv(ROOT / "eval" / "questions.csv")
    write_table("questions_table", ["\\#", "Type", "Question", "Source"],
                [[str(r.id), r.type, tex_escape(r.question),
                  tex_escape(str(r.source).replace("airbus_", "A:").replace("boeing_", "B:")
                             .replace("aircraft_characteristics", "AC").replace("airport_planning", "ACAP")
                             .replace("statistical_summary_accidents", "statsum").replace("; ", ", ")
                             if isinstance(r.source, str) else "--")]
                 for r in qs.itertuples()],
                "rlp{7.2cm}p{3.4cm}", "The evaluation questions (A: = Airbus, B: = Boeing document).", "tab:questions")

    # ---------- dialogue test (appendix) ----------
    dfile = ROOT / "eval" / "dialogue_test.csv"
    if dfile.exists():
        d = pd.read_csv(dfile)
        write_table("dialogue_table", ["Dlg.", "Turn", "User message", "Rewritten standalone question"],
                    [[str(r.dialogue), str(r.turn), tex_escape(r.user_message), tex_escape(r.rewritten_question)]
                     for r in d.itertuples()], "rrp{4.6cm}p{6.4cm}",
                    "Follow-up questions and the standalone questions produced by the dialogue system.", "tab:dialogue")
    else:
        write_table("dialogue_table", ["Dlg.", "Turn", "User message", "Rewritten standalone question"], [],
                    "rrp{4.6cm}p{6.4cm}",
                    "Follow-up questions and the standalone questions produced by the dialogue system.", "tab:dialogue")

    # ---------- numbers for the text ----------
    def macro(k):
        return "".join(ch for ch in k if ch.isalpha())
    lines = [f"\\newcommand{{\\num{macro(k)}}}{{{tex_escape(v)}}}" for k, v in numbers.items()]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"indexing": idx, "query": q, "numbers": numbers}
    (METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
