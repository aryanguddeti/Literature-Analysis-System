"""
output/timeline.py
──────────────────
Format 3: Plotly interactive paradigm shift timeline.
Owned by Devang.

Generates an interactive Plotly figure showing the research field's
evolution over time. Returns a Plotly Figure object that Streamlit
renders with st.plotly_chart().

Visual design:
  - X axis: year
  - Y axis: impact level (incremental → significant → paradigm-shift)
  - Bubble size: citation count of the key paper
  - Color: methodology cluster
  - Hover: full shift description + key paper
  - Annotations: for paradigm-shift events only
"""

import json
from typing import Optional

import plotly.graph_objects as go
import plotly.express as px

from core.context import SharedContext

# Impact level → numeric Y position
_IMPACT_RANK = {
    "incremental": 1,
    "significant": 2,
    "paradigm-shift": 3,
}

_IMPACT_LABELS = {
    1: "Incremental",
    2: "Significant",
    3: "Paradigm Shift",
}

# Color palette for clusters
_CLUSTER_COLORS = [
    "#4C9BE8", "#E8674C", "#4CE8A0", "#E8C44C",
    "#A04CE8", "#E84CA0", "#4CE8D4", "#E8834C",
]


def _get_citation_count(key_paper_title: str, context: SharedContext) -> int:
    """Look up citation count for a paper by title (fuzzy match)."""
    title_lower = key_paper_title.lower()
    for paper in context.user_confirmed_papers:
        if paper.get("title", "").lower() == title_lower:
            return paper.get("citationCount", 0)
    # Partial match fallback
    for paper in context.user_confirmed_papers:
        if title_lower[:30] in paper.get("title", "").lower():
            return paper.get("citationCount", 0)
    return 0


def generate(context: SharedContext) -> go.Figure:
    """
    Generate an interactive Plotly timeline of paradigm shifts.

    Args:
        context: SharedContext with paradigm_timeline and user_confirmed_papers.

    Returns:
        Plotly Figure object. Use st.plotly_chart(fig, use_container_width=True).
    """
    timeline = context.paradigm_timeline

    if not timeline:
        # Return empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No paradigm shifts identified for this topic.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666"),
        )
        fig.update_layout(title=f"Paradigm Timeline: {context.topic}")
        return fig

    # ── Prepare data ──────────────────────────────────────────────────────────
    years, y_positions, sizes, colors, hovers, labels = [], [], [], [], [], []
    impact_texts = []
    cluster_list = []

    # Get unique clusters for color mapping
    all_clusters = list(set(e.get("affected_cluster", "other") for e in timeline))
    cluster_color_map = {
        cluster: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        for i, cluster in enumerate(all_clusters)
    }

    for event in timeline:
        year = event.get("year")
        if not year:
            continue

        impact = event.get("impact", "incremental")
        y_pos = _IMPACT_RANK.get(impact, 1)
        cluster = event.get("affected_cluster", "other")
        citations = _get_citation_count(event.get("key_paper", ""), context)
        size = max(15, min(60, 15 + citations / 500))  # Scale bubble size

        years.append(year)
        y_positions.append(y_pos)
        sizes.append(size)
        colors.append(cluster_color_map.get(cluster, "#4C9BE8"))
        cluster_list.append(cluster)
        impact_texts.append(impact)

        hover_text = (
            f"<b>{year}</b><br>"
            f"<b>Impact:</b> {impact}<br>"
            f"<b>Cluster:</b> {cluster}<br>"
            f"<b>Key Paper:</b> {event.get('key_paper', '—')}<br>"
            f"<b>Citations:</b> {citations:,}<br><br>"
            f"{event.get('shift_description', '')}"
        )
        hovers.append(hover_text)

        label = event.get("key_paper", "")[:40] if impact == "paradigm-shift" else ""
        labels.append(label)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Background bands for impact levels
    for rank, label in _IMPACT_LABELS.items():
        fig.add_hrect(
            y0=rank - 0.4, y1=rank + 0.4,
            fillcolor=["#F0F4FF", "#FFF8F0", "#FFF0F0"][rank - 1],
            opacity=0.4,
            line_width=0,
            annotation_text=label,
            annotation_position="left",
            annotation_font=dict(size=11, color="#888"),
        )

    # Plot each cluster as a separate trace (for legend)
    for cluster in all_clusters:
        cluster_indices = [i for i, c in enumerate(cluster_list) if c == cluster]
        if not cluster_indices:
            continue

        fig.add_trace(go.Scatter(
            x=[years[i] for i in cluster_indices],
            y=[y_positions[i] for i in cluster_indices],
            mode="markers+text",
            name=cluster,
            marker=dict(
                size=[sizes[i] for i in cluster_indices],
                color=cluster_color_map[cluster],
                line=dict(width=2, color="white"),
                opacity=0.85,
            ),
            text=[labels[i] for i in cluster_indices],
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            hovertext=[hovers[i] for i in cluster_indices],
            hoverinfo="text",
        ))

    # Connecting timeline line
    sorted_events = sorted(zip(years, y_positions), key=lambda x: x[0])
    if sorted_events:
        sorted_years, sorted_y = zip(*sorted_events)
        fig.add_trace(go.Scatter(
            x=sorted_years,
            y=sorted_y,
            mode="lines",
            line=dict(color="#CCCCCC", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    min_year = min(years) - 1 if years else 2000
    max_year = max(years) + 1 if years else 2025

    fig.update_layout(
        title=dict(
            text=f"Research Paradigm Timeline: {context.topic}",
            font=dict(size=18, color="#1a1a2e"),
            x=0.0,
        ),
        xaxis=dict(
            title="Year",
            range=[min_year, max_year],
            dtick=1,
            showgrid=True,
            gridcolor="#EEEEEE",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Impact Level",
            range=[0.3, 3.7],
            tickvals=[1, 2, 3],
            ticktext=["Incremental", "Significant", "Paradigm Shift"],
            showgrid=False,
            tickfont=dict(size=12),
        ),
        legend=dict(
            title="Methodology Cluster",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        hovermode="closest",
        margin=dict(l=100, r=40, t=80, b=60),
    )

    return fig


def generate_papers_scatter(context: SharedContext) -> go.Figure:
    """
    Secondary visualization: scatter plot of all papers by year vs citation count,
    colored by methodology cluster.

    Useful as a companion to the timeline view.
    """
    papers = context.user_confirmed_papers
    extractions = context.extracted_knowledge
    extraction_map = {e.get("paperId", ""): e for e in extractions}

    if not papers:
        return go.Figure()

    all_clusters = list(set(
        extraction_map.get(p.get("paperId") or p.get("arxiv_id", ""), {}).get("methodology_cluster", "other")
        for p in papers
    ))
    cluster_color_map = {
        c: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        for i, c in enumerate(all_clusters)
    }

    fig = go.Figure()

    for cluster in all_clusters:
        cluster_papers = [
            p for p in papers
            if extraction_map.get(p.get("paperId") or p.get("arxiv_id", ""), {}).get("methodology_cluster", "other") == cluster
        ]
        if not cluster_papers:
            continue

        fig.add_trace(go.Scatter(
            x=[p.get("year") for p in cluster_papers],
            y=[p.get("citationCount", 0) for p in cluster_papers],
            mode="markers",
            name=cluster,
            marker=dict(
                size=[max(8, min(30, p.get("relevance_score", 0) * 40)) for p in cluster_papers],
                color=cluster_color_map[cluster],
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            hovertext=[
                f"<b>{p.get('title', '?')}</b><br>"
                f"Year: {p.get('year', 'N/A')}<br>"
                f"Citations: {p.get('citationCount', 0):,}<br>"
                f"Relevance: {p.get('relevance_score', 0):.3f}"
                for p in cluster_papers
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"Paper Landscape: {context.topic}",
        xaxis_title="Year",
        yaxis_title="Citation Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        hovermode="closest",
        legend=dict(title="Cluster", orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )

    return fig
