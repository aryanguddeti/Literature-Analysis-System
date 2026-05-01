"""
output/knowledge_map.py
───────────────────────
Format 2: Citation / Knowledge Network Graph.
Owned by Devang.

Generates a Plotly-based network graph (primary) and exports the data
as JSON for the D3.js HTML template (ui/templates/knowledge_graph.html).

Network nodes = papers
Network edges = citation relationships (paper A cites paper B)
Node color    = methodology cluster
Node size     = citation count (log-scaled)
Edge thickness = shared cluster (thicker = same cluster)

Fallback: If D3 is too complex or blocks progress, the Plotly version
is used directly in Streamlit (as documented in the risk register).
"""

import json
import math
from typing import Optional

import plotly.graph_objects as go
import networkx as nx

from core.context import SharedContext

_CLUSTER_COLORS = {
    "transformer-based": "#4C9BE8",
    "reinforcement-learning": "#E8674C",
    "GNN-based": "#4CE8A0",
    "CNN-based": "#E8C44C",
    "RNN-based": "#A04CE8",
    "optimization": "#E84CA0",
    "probabilistic": "#4CE8D4",
    "hybrid": "#E8834C",
    "survey": "#8B9E8E",
    "other": "#AAAAAA",
}

_DEFAULT_COLOR = "#AAAAAA"


def _log_scale_size(citation_count: int, min_size: float = 10, max_size: float = 50) -> float:
    """Map citation count to node size using log scale."""
    if citation_count <= 0:
        return min_size
    scaled = math.log1p(citation_count) * 5
    return max(min_size, min(max_size, scaled))


def _build_graph(context: SharedContext) -> nx.DiGraph:
    """Build a directed graph from paper citation relationships."""
    papers = context.user_confirmed_papers
    extractions = context.extracted_knowledge
    extraction_map = {e.get("paperId", ""): e for e in extractions}

    # Build paperId set for edge filtering
    paper_ids = set(
        p.get("paperId") or p.get("arxiv_id", "")
        for p in papers
    )

    G = nx.DiGraph()

    # Add nodes
    for paper in papers:
        pid = paper.get("paperId") or paper.get("arxiv_id", "")
        ext = extraction_map.get(pid, {})
        G.add_node(
            pid,
            title=paper.get("title", "Untitled"),
            year=paper.get("year"),
            authors=paper.get("authors", []),
            citations=paper.get("citationCount", 0),
            relevance=paper.get("relevance_score", 0.0),
            cluster=ext.get("methodology_cluster", "other"),
            key_finding=ext.get("key_finding", ""),
            pdf_url=paper.get("pdf_url"),
        )

    # Add edges (citations — paper cites another paper in our set)
    for paper in papers:
        pid = paper.get("paperId") or paper.get("arxiv_id", "")
        for ref_id in paper.get("references", []):
            if ref_id in paper_ids and ref_id != pid:
                G.add_edge(pid, ref_id)  # pid cites ref_id

    return G


def generate(context: SharedContext) -> go.Figure:
    """
    Generate a Plotly network graph of paper citation relationships.

    Args:
        context: SharedContext with user_confirmed_papers and extracted_knowledge.

    Returns:
        Plotly Figure with the knowledge network.
    """
    if not context.user_confirmed_papers:
        fig = go.Figure()
        fig.add_annotation(text="No papers available.", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    G = _build_graph(context)

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="Could not build citation graph.", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    # Layout using spring layout (force-directed)
    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = {node: (i % 5, i // 5) for i, node in enumerate(G.nodes())}

    # ── Edge traces ───────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos.get(edge[0], (0, 0))
        x1, y1 = pos.get(edge[1], (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#CCCCCC"),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Node traces per cluster ────────────────────────────────────────────────
    clusters = list(set(G.nodes[n].get("cluster", "other") for n in G.nodes()))
    node_traces = []

    for cluster in clusters:
        cluster_nodes = [n for n in G.nodes() if G.nodes[n].get("cluster", "other") == cluster]
        if not cluster_nodes:
            continue

        node_x = [pos[n][0] for n in cluster_nodes]
        node_y = [pos[n][1] for n in cluster_nodes]
        node_sizes = [_log_scale_size(G.nodes[n].get("citations", 0)) for n in cluster_nodes]
        node_color = _CLUSTER_COLORS.get(cluster, _DEFAULT_COLOR)

        hover_texts = []
        for n in cluster_nodes:
            d = G.nodes[n]
            authors = d.get("authors", [])
            author_str = authors[0].split()[-1] if authors else "Unknown"
            in_deg = G.in_degree(n)
            out_deg = G.out_degree(n)
            hover_texts.append(
                f"<b>{d.get('title', '?')}</b><br>"
                f"{author_str} et al., {d.get('year', 'N/A')}<br>"
                f"Citations: {d.get('citations', 0):,}<br>"
                f"Cluster: {cluster}<br>"
                f"Cited by (in graph): {in_deg} · Cites: {out_deg}<br>"
                f"Relevance: {d.get('relevance', 0.0):.3f}<br><br>"
                f"{d.get('key_finding', '')[:120]}"
            )

        labels = [G.nodes[n].get("title", "")[:25] for n in cluster_nodes]

        node_traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            name=cluster,
            marker=dict(
                size=node_sizes,
                color=node_color,
                line=dict(width=2, color="white"),
                opacity=0.85,
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color="#444"),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)

    fig.update_layout(
        title=dict(
            text=f"Citation Network: {context.topic}",
            font=dict(size=18, color="#1a1a2e"),
        ),
        showlegend=True,
        legend=dict(title="Cluster", orientation="v", x=1.01, y=0.5),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        margin=dict(l=20, r=120, t=60, b=20),
    )

    return fig


def export_d3_json(context: SharedContext) -> dict:
    """
    Export the graph as a D3-compatible JSON object.

    Format:
        {
          "nodes": [{"id", "title", "year", "cluster", "citations", "relevance"}, ...],
          "links": [{"source", "target"}, ...]
        }

    Used by ui/templates/knowledge_graph.html.
    """
    G = _build_graph(context)

    nodes = []
    for node_id in G.nodes():
        d = G.nodes[node_id]
        authors = d.get("authors", [])
        nodes.append({
            "id": node_id,
            "title": d.get("title", "Untitled"),
            "year": d.get("year"),
            "authors": authors[:2],
            "cluster": d.get("cluster", "other"),
            "citations": d.get("citations", 0),
            "relevance": round(d.get("relevance", 0.0), 3),
            "key_finding": d.get("key_finding", "")[:200],
            "pdf_url": d.get("pdf_url"),
            "color": _CLUSTER_COLORS.get(d.get("cluster", "other"), _DEFAULT_COLOR),
        })

    links = [{"source": u, "target": v} for u, v in G.edges()]

    return {"nodes": nodes, "links": links}
