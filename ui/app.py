"""
ui/app.py
─────────
Streamlit web UI for the Literature Analysis System.
Owned by Devang.

Screens / flow:
  1. Topic Input       — user enters research topic, selects output format
  2. Retrieval Status  — live progress while Retriever runs
  3. HITL Checkpoint   — user reviews and confirms papers before analysis
  4. Analysis Status   — live progress while Analyst runs
  5. Output Display    — shows the generated output with download options

Session state keys:
  st.session_state.context         — SharedContext object
  st.session_state.stage           — UI stage name (string)
  st.session_state.retriever_done  — bool flag
  st.session_state.analyst_done    — bool flag
  st.session_state.showcase_done   — bool flag
  st.session_state.error_message   — string | None
"""

import os
import sys
import time
import threading
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from core.context import SharedContext
from core.hitl import get_display_papers, confirm_papers
import agents.retriever as retriever_agent
import agents.analyst as analyst_agent
import agents.showcase as showcase_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Literature Analysis System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #1a1a2e;
        margin-bottom: 0;
        line-height: 1.1;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-top: 0.4rem;
        margin-bottom: 2rem;
    }

    .stage-badge {
        display: inline-block;
        background: #E8F4FD;
        color: #1565C0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    .paper-card {
        background: white;
        border: 1px solid #E8EDF2;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.2s;
    }

    .paper-card:hover {
        border-color: #4C9BE8;
    }

    .paper-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1a1a2e;
    }

    .paper-meta {
        font-size: 0.8rem;
        color: #888;
        margin-top: 2px;
    }

    .relevance-bar-bg {
        background: #F0F4FF;
        border-radius: 4px;
        height: 6px;
        margin-top: 6px;
    }

    .stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1565C0;
        line-height: 1;
    }

    .stat-label {
        font-size: 0.78rem;
        color: #888;
        margin-top: 4px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }

    .output-section {
        background: white;
        border: 1px solid #E8EDF2;
        border-radius: 12px;
        padding: 2rem;
    }

    div[data-testid="stProgress"] > div {
        background-color: #4C9BE8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state():
    defaults = {
        "context": None,
        "stage": "input",
        "retriever_done": False,
        "analyst_done": False,
        "showcase_done": False,
        "error_message": None,
        "selected_paper_ids": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Background thread runners ─────────────────────────────────────────────────

def _run_retriever(context: SharedContext):
    try:
        retriever_agent.run(context)
    except Exception as exc:
        context.pipeline_status = "error"
        context.errors.append({
            "stage": "retriever_thread",
            "message": str(exc),
            "recoverable": False,
        })
    st.session_state.retriever_done = True


def _run_analyst(context: SharedContext):
    try:
        analyst_agent.run(context)
    except Exception as exc:
        context.pipeline_status = "error"
        context.errors.append({
            "stage": "analyst_thread",
            "message": str(exc),
            "recoverable": False,
        })
    st.session_state.analyst_done = True


def _run_showcase(context: SharedContext):
    try:
        showcase_agent.run(context)
    except Exception as exc:
        context.pipeline_status = "error"
        context.errors.append({
            "stage": "showcase_thread",
            "message": str(exc),
            "recoverable": False,
        })
    st.session_state.showcase_done = True


# ── Helper: reset ─────────────────────────────────────────────────────────────

def _reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()
    st.rerun()


# ── UI Components ─────────────────────────────────────────────────────────────

def _header(subtitle: str = ""):
    st.markdown('<p class="main-title">📚 Literature Analysis System</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)


def _error_banner(message: str):
    st.error(f"**Error:** {message}")
    if st.button("↩ Start Over"):
        _reset()


def _stage_badge(label: str):
    st.markdown(f'<span class="stage-badge">{label}</span>', unsafe_allow_html=True)


# ── Screen 1: Topic Input ─────────────────────────────────────────────────────

def screen_input():
    _header("Autonomous literature discovery, analysis & synthesis powered by multi-agent AI.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### What would you like to research?")
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. attention mechanisms in NLP, graph neural networks, RLHF",
            label_visibility="collapsed",
        )

        st.markdown("#### Output format")
        format_options = {
            "📋 Structured Briefing": "briefing",
            "📝 Written Review Essay": "written_review",
            "🕸️ Citation Network Graph": "knowledge_map",
            "📈 Paradigm Timeline": "timeline",
        }
        format_labels = list(format_options.keys())
        selected_label = st.radio(
            "Select output format",
            format_labels,
            label_visibility="collapsed",
            horizontal=True,
        )
        selected_format = format_options[selected_label]

        st.markdown("")
        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
            if not topic.strip():
                st.warning("Please enter a research topic.")
            else:
                ctx = SharedContext(topic=topic.strip(), output_format=selected_format)
                st.session_state.context = ctx
                st.session_state.stage = "retrieving"
                st.session_state.retriever_done = False
                # Start retriever in background
                t = threading.Thread(target=_run_retriever, args=(ctx,), daemon=True)
                t.start()
                st.rerun()

    with col2:
        st.markdown("#### How it works")
        steps = [
            ("🔍", "Retrieve", "Semantic Scholar + ArXiv search across multiple query variants"),
            ("✋", "Review", "You confirm which papers to analyze"),
            ("🧠", "Analyze", "Gemini extracts knowledge, finds contradictions & timeline"),
            ("✨", "Showcase", "Choose your output: briefing, review, graph, or timeline"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.markdown("")


# ── Screen 2: Retrieval Progress ─────────────────────────────────────────────

def screen_retrieving():
    ctx: SharedContext = st.session_state.context
    _header(f'Searching for papers on: *"{ctx.topic}"*')
    _stage_badge("Step 1 of 4 — Retrieving Papers")

    if ctx.pipeline_status == "error":
        errors = [e for e in ctx.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "Unknown retrieval error."
        _error_banner(msg)
        return

    # Progress display
    progress_ph = st.empty()
    status_ph = st.empty()
    detail_ph = st.empty()

    status_messages = [
        "Generating query variants with Gemini…",
        "Searching Semantic Scholar…",
        "Citation snowballing…",
        "Enriching with ArXiv metadata…",
        "Deduplicating and scoring relevance…",
    ]

    # Poll pipeline_status — works across threads
    if ctx.pipeline_status in ("awaiting_hitl", "analysing", "done") or len(ctx.deduplicated_papers) > 0:
        st.session_state.stage = "hitl"
        st.rerun()
        return

    if ctx.pipeline_status == "error":
        errors = [e for e in ctx.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "Retrieval failed."
        _error_banner(msg)
        return

    i = st.session_state.get("retriever_poll_i", 0)
    progress = min(0.9, (i % 50) / 50)
    progress_ph.progress(progress)
    status_ph.markdown(f"**{status_messages[i % len(status_messages)]}**")
    if ctx.query_variants:
        detail_ph.caption("Queries: " + " · ".join(ctx.query_variants[:3]))
    st.session_state.retriever_poll_i = i + 1

    if st.button("🔄 Retrieval done? Click to continue"):
        st.rerun()
    time.sleep(3)
    st.rerun()


# ── Screen 3: HITL Checkpoint ─────────────────────────────────────────────────

def screen_hitl():
    ctx: SharedContext = st.session_state.context
    papers = get_display_papers(ctx)

    _header(f'"{ctx.topic}"')
    _stage_badge("Step 2 of 4 — Review Papers")

    st.markdown(
        f"The Retriever found **{len(papers)} papers**. "
        "Select which ones to include in the analysis. "
        "Papers are ranked by relevance score."
    )

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    years = [p["year"] for p in papers if p["year"] != "N/A"]
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(papers)}</div><div class="stat-label">Papers Found</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{min(years) if years else "—"}</div><div class="stat-label">Earliest Year</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{max(years) if years else "—"}</div><div class="stat-label">Latest Year</div></div>', unsafe_allow_html=True)
    with c4:
        avg_relevance = sum(p["relevance_score"] for p in papers) / len(papers) if papers else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{avg_relevance:.2f}</div><div class="stat-label">Avg Relevance</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Select All / Deselect All
    col_a, col_b, _ = st.columns([1, 1, 5])
    if col_a.button("✅ Select All"):
        st.session_state.selected_paper_ids = [p["paperId"] for p in papers]
    if col_b.button("☐ Deselect All"):
        st.session_state.selected_paper_ids = []

    if "selected_paper_ids" not in st.session_state:
        st.session_state.selected_paper_ids = [p["paperId"] for p in papers]

    selected_ids = set(st.session_state.selected_paper_ids)
    new_selected = []

    # Paper list
    for paper in papers:
        pid = paper["paperId"]
        is_checked = pid in selected_ids

        col_check, col_content = st.columns([0.05, 0.95])
        with col_check:
            checked = st.checkbox("", value=is_checked, key=f"paper_{pid}", label_visibility="collapsed")
        with col_content:
            relevance_pct = int(paper["relevance_score"] * 100)
            pdf_link = f' · [PDF]({paper["pdf_url"]})' if paper.get("pdf_url") else ""
            st.markdown(
                f'<div class="paper-card">'
                f'<div class="paper-title">{paper["title"]}</div>'
                f'<div class="paper-meta">{paper["authors"]} · {paper["year"]} · '
                f'{paper["citationCount"]:,} citations · {paper["source"]}{pdf_link}</div>'
                f'<div class="relevance-bar-bg"><div style="width:{relevance_pct}%;background:#4C9BE8;height:6px;border-radius:4px;"></div></div>'
                f'<div class="paper-meta" style="margin-top:3px;">Relevance: {paper["relevance_score"]:.3f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Abstract"):
                st.caption(paper["abstract_short"] or "*No abstract available.*")

        if checked:
            new_selected.append(pid)

    st.session_state.selected_paper_ids = new_selected
    st.markdown("---")

    n_selected = len(new_selected)
    confirm_col, _ = st.columns([1, 2])
    with confirm_col:
        btn_label = f"✅ Confirm {n_selected} papers & Start Analysis" if n_selected > 0 else "Select at least 1 paper"
        if st.button(btn_label, type="primary", disabled=(n_selected == 0), use_container_width=True):
            confirm_papers(ctx, new_selected)
            st.session_state.stage = "analysing"
            st.session_state.analyst_done = False
            t = threading.Thread(target=_run_analyst, args=(ctx,), daemon=True)
            t.start()
            st.rerun()


# ── Screen 4: Analysis Progress ───────────────────────────────────────────────

def screen_analysing():
    ctx: SharedContext = st.session_state.context
    _header(f'Analyzing {len(ctx.user_confirmed_papers)} papers on: *"{ctx.topic}"*')
    _stage_badge("Step 3 of 4 — Analyzing")

    if ctx.pipeline_status == "error":
        errors = [e for e in ctx.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "Analysis failed."
        _error_banner(msg)
        return

    progress_ph = st.empty()
    status_ph = st.empty()

    passes = [
        (0.1, "Pass 1/4: Extracting methodology & findings from each paper…"),
        (0.4, "Pass 2/4: Clustering papers by methodology…"),
        (0.65, "Pass 3/4: Detecting contradictions within clusters…"),
        (0.85, "Pass 4/4: Building paradigm shift timeline…"),
        (0.95, "Generating your output…"),
    ]

    # Check if analyst is done by inspecting context directly
    if ctx.pipeline_status in ("showcasing", "done") or len(ctx.extracted_knowledge) > 0:
        progress_ph.progress(1.0)
        if ctx.pipeline_status == "error":
            errors = [e for e in ctx.errors if not e.get("recoverable")]
            msg = errors[-1]["message"] if errors else "Analysis failed."
            _error_banner(msg)
            return
        if not st.session_state.get("showcase_started"):
            st.session_state.showcase_started = True
            st.session_state.showcase_done = False
            t = threading.Thread(target=_run_showcase, args=(ctx,), daemon=True)
            t.start()
        st.session_state.stage = "showcasing"
        st.rerun()
        return

    if ctx.pipeline_status == "error":
        errors = [e for e in ctx.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "Analysis failed."
        _error_banner(msg)
        return

    i = st.session_state.get("analyst_poll_i", 0)
    pass_idx = min(i // 15, len(passes) - 1)
    progress, msg = passes[pass_idx]
    progress_ph.progress(progress)
    status_ph.markdown(f"**{msg}**")
    st.session_state.analyst_poll_i = i + 1

    if st.button("🔄 Analysis done? Click to continue"):
        st.rerun()
    time.sleep(3)
    st.rerun()


# ── Screen 5: Showcasing (brief spinner) ─────────────────────────────────────

def screen_showcasing():
    ctx: SharedContext = st.session_state.context
    _header("Generating your output…")
    _stage_badge("Step 4 of 4 — Showcase")

    progress_ph = st.empty()

    if ctx.final_output is not None or st.session_state.get("showcase_done"):
        progress_ph.progress(1.0)
        st.session_state.stage = "output"
        st.rerun()
        return

    if ctx.pipeline_status == "error":
        st.error("Showcase failed.")
        return

    i = st.session_state.get("showcase_poll_i", 0)
    progress_ph.progress(min(0.95, 0.3 + (i % 20) * 0.03))
    st.session_state.showcase_poll_i = i + 1

    if st.button("🔄 Output ready? Click to continue"):
        st.rerun()
    time.sleep(2)
    st.rerun()


# ── Screen 6: Output Display ──────────────────────────────────────────────────

def screen_output():
    ctx: SharedContext = st.session_state.context

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 📊 Analysis Summary")
        st.metric("Papers analyzed", len(ctx.user_confirmed_papers))
        st.metric("Contradictions", len(ctx.contradiction_report))
        st.metric("Paradigm shifts", len(ctx.paradigm_timeline))

        major = len([c for c in ctx.contradiction_report if c.get("severity") == "major"])
        if major:
            st.warning(f"⚡ {major} major contradiction(s) found")

        st.markdown("---")
        st.markdown("### 📥 Export")

        md_content = showcase_agent.export_as_markdown(ctx)
        st.download_button(
            "⬇️ Download Markdown",
            data=md_content,
            file_name=f"literature_review_{ctx.topic[:30].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        json_content = showcase_agent.export_as_json(ctx)
        st.download_button(
            "⬇️ Download JSON",
            data=json_content,
            file_name=f"literature_analysis_{ctx.topic[:30].replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown("---")

        # Switch output format
        st.markdown("### 🔄 Change Format")
        format_options = {
            "📋 Briefing": "briefing",
            "📝 Review Essay": "written_review",
            "🕸️ Citation Graph": "knowledge_map",
            "📈 Timeline": "timeline",
        }
        new_format_label = st.selectbox(
            "Output format",
            list(format_options.keys()),
            label_visibility="collapsed",
        )
        if st.button("Generate", use_container_width=True):
            ctx.output_format = format_options[new_format_label]
            ctx.final_output = None
            st.session_state.stage = "showcasing"
            st.session_state.showcase_done = False
            t = threading.Thread(target=_run_showcase, args=(ctx,), daemon=True)
            t.start()
            st.rerun()

        st.markdown("---")
        if st.button("🔁 New Search", use_container_width=True):
            _reset()

    # ── Main output area ──────────────────────────────────────────────────────
    _header(f'"{ctx.topic}"')
    _stage_badge(f"Output · {ctx.output_format.replace('_', ' ').title()}")

    if ctx.pipeline_status == "error" and not ctx.final_output:
        errors = [e for e in ctx.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "An error occurred."
        st.error(f"**Output generation failed:** {msg}")
        return

    # Non-recoverable errors in sidebar
    non_recoverable = [e for e in ctx.errors if not e.get("recoverable")]
    if non_recoverable:
        with st.expander("⚠️ Pipeline warnings"):
            for e in ctx.errors:
                icon = "🔴" if not e.get("recoverable") else "🟡"
                st.caption(f"{icon} [{e['stage']}] {e['message']}")

    output = ctx.final_output

    # ── Format-specific rendering ─────────────────────────────────────────────
    if ctx.output_format in ("briefing", "written_review"):
        # Markdown text output
        if isinstance(output, str):
            st.markdown(output)
        else:
            st.write(output)

    elif ctx.output_format == "timeline":
        # Plotly timeline chart
        col1, col2 = st.columns([3, 1])
        with col1:
            if output is not None:
                st.plotly_chart(output, use_container_width=True)
        with col2:
            st.markdown("#### Paradigm Events")
            for event in ctx.paradigm_timeline:
                impact = event.get("impact", "incremental")
                icon = {"paradigm-shift": "🔴", "significant": "🟡", "incremental": "🟢"}.get(impact, "⚪")
                st.markdown(f"**{event.get('year', '?')}** {icon}")
                st.caption(event.get("shift_description", "")[:100])

        # Paper scatter plot
        st.markdown("#### Paper Landscape")
        from output.timeline import generate_papers_scatter
        scatter_fig = generate_papers_scatter(ctx)
        st.plotly_chart(scatter_fig, use_container_width=True)

    elif ctx.output_format == "knowledge_map":
        # Plotly network graph + D3 JSON option
        if output is not None:
            st.plotly_chart(output, use_container_width=True)

        st.markdown("#### Network Statistics")
        from output.knowledge_map import export_d3_json
        graph_data = export_d3_json(ctx)
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes (papers)", len(graph_data["nodes"]))
        c2.metric("Edges (citations)", len(graph_data["links"]))
        c3.metric("Clusters", len(set(n["cluster"] for n in graph_data["nodes"])))

        with st.expander("📄 D3.js JSON data"):
            st.code(json.dumps(graph_data, indent=2)[:3000] + "\n...", language="json")
            st.download_button(
                "⬇️ Download D3 JSON",
                data=json.dumps(graph_data, indent=2),
                file_name=f"knowledge_graph_{ctx.topic[:20].replace(' ', '_')}.json",
                mime="application/json",
            )


# ── Router ────────────────────────────────────────────────────────────────────

SCREENS = {
    "input":      screen_input,
    "retrieving": screen_retrieving,
    "hitl":       screen_hitl,
    "analysing":  screen_analysing,
    "showcasing": screen_showcasing,
    "output":     screen_output,
}

def main():
    stage = st.session_state.get("stage", "input")
    screen_fn = SCREENS.get(stage, screen_input)
    screen_fn()


if __name__ == "__main__":
    main()
