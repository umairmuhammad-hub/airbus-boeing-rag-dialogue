"""Interactive pictures of the two knowledge graphs (for the presentation).

Run:  python show_graph.py      -> opens graph_lightrag.html and graph_graphrag.html in the browser
"""
import webbrowser

import networkx as nx
from pyvis.network import Network

from config import GRAPHRAG_DIR, LIGHTRAG_DIR, ROOT

GRAPHS = {"lightrag": LIGHTRAG_DIR / "graph_chunk_entity_relation.graphml",
          "graphrag": GRAPHRAG_DIR / "output" / "graph.graphml"}

for name, path in GRAPHS.items():
    if not path.exists():
        print(f"{name}: no graph yet ({path.name} missing)")
        continue
    G = nx.read_graphml(path).to_undirected()
    keep = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:200]]
    H = G.subgraph(keep)
    net = Network(height="900px", width="100%", notebook=False, cdn_resources="in_line")
    for n, data in H.nodes(data=True):
        net.add_node(str(n), label=str(n)[:30], title=str(data.get("description", ""))[:500],
                     size=8 + 2 * H.degree[n])
    for a, b, data in H.edges(data=True):
        net.add_edge(str(a), str(b), title=str(data.get("description", ""))[:300])
    out = ROOT / f"graph_{name}.html"
    net.write_html(str(out))
    print(f"{name}: {G.number_of_nodes()} entities, {G.number_of_edges()} relations -> {out.name}")
    webbrowser.open(out.as_uri())
