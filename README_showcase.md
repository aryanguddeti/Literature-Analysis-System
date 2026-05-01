# Showcase Agent & UI — README
**Owner: Devang Chaturvedi (P3)**

---

## Overview

The Showcase Agent is the final stage of the Literature Analysis System pipeline. It takes the structured knowledge produced by the Analyst and transforms it into human-readable outputs through a polished Streamlit web interface.

---

## Files

| File | Description |
|------|-------------|
| `agents/showcase.py` | Main showcase agent — dispatches to output formats |
| `core/hitl.py` | Human-in-the-Loop checkpoint logic |
| `ui/app.py` | Full Streamlit web application |
| `ui/templates/knowledge_graph.html` | Standalone D3.js citation network graph |
| `output/briefing.py` | Format 4: Structured bullet briefing |
| `output/written_review.py` | Format 1: Written synthesis essay |
| `output/timeline.py` | Format 3: Plotly paradigm shift timeline |
| `output/knowledge_map.py` | Format 2: Citation network graph |
| `output/pdf_export.py` | PDF export for all formats |
| `eval/benchmark_runner.py` | Automated evaluation benchmark runner |
| `main.py` | CLI orchestrator wiring all 3 agents |

---

## Running the App

### Web UI (recommended)
```bash
python -m streamlit run ui/app.py
```
Opens at `http://localhost:8501`

### CLI
```bash
python main.py --topic "attention mechanisms in NLP"
python main.py --topic "RLHF" --format written_review
python main.py --topic "graph neural networks" --no-hitl
```

### Live deployment
The app is deployed at Streamlit Community Cloud — share the link with the team for demo day.

---

## Pipeline Flow

```
User types topic
      ↓
[Retriever] searches Semantic Scholar + ArXiv
      ↓
[HITL] user confirms which papers to analyze
      ↓
[Analyst] extracts knowledge, finds contradictions, builds timeline
      ↓
[Showcase] generates chosen output format
      ↓
User downloads PDF or JSON
```

---

## Output Formats

### Format 1 — Written Review Essay
A full literature review essay written by Gemini covering methodology clusters, contradictions, historical trajectory, and future directions. Reads like a human-written survey paper.

### Format 2 — Citation Network Graph
Interactive force-directed graph showing citation relationships between papers. Nodes colored by methodology cluster, sized by citation count. Built with Plotly (in-app) and D3.js (standalone HTML).

### Format 3 — Paradigm Timeline
Interactive Plotly chart showing how the research field evolved over time. X-axis = year, Y-axis = impact level (incremental → significant → paradigm-shift).

### Format 4 — Structured Briefing (Default)
Executive summary + top-N papers with 3-line summaries + contradiction highlights + paradigm shift highlights. The Phase 1 deliverable — simplest and fastest to generate.

---

## HITL Checkpoint

The Human-in-the-Loop checkpoint pauses the pipeline between retrieval and analysis so the user can review and select which papers to include.

**Implementation:** Uses Streamlit `session_state` flags (not `st.stop()`) to avoid timing issues.

**Key functions in `core/hitl.py`:**
- `confirm_papers(context, selected_ids)` — records user selection
- `auto_confirm_all(context)` — headless mode for CLI/testing
- `get_display_papers(context)` — formats papers for the UI checklist

---

## Export

All output formats support:
- **Markdown download** — `.md` file of the text output
- **JSON download** — full structured data (papers, extractions, contradictions, timeline)
- **PDF download** — formatted PDF via `fpdf2`

---

## Evaluation — Synthesis Quality Rubric

Devang's evaluation metric scores output quality on 5 dimensions (each 0–5):

| Dimension | What it measures |
|-----------|-----------------|
| Thematic organization | Output organized by methodology cluster |
| Citation accuracy | Papers cited correctly with year/author |
| Contradiction coverage | Contradictions mentioned and explained |
| Clarity | Writing is clear and readable |
| Completeness | Covers all major papers in the confirmed set |

**Target: ≥ 4.0/5.0 average** across all 3 benchmark topics.

### Running the benchmark
```bash
python eval/benchmark_runner.py --all --save-results
python eval/benchmark_runner.py --topic attention_mechanisms
```

---

## Dependencies (Showcase-specific)

```
streamlit>=1.35.0
plotly>=5.22.0
networkx>=3.3
fpdf2>=2.7.0
```

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| D3.js graph too complex | Plotly fallback already implemented |
| Streamlit HITL timing issues | Used `session_state` flags instead of `st.stop()` |
| Output generation fails | Automatic fallback to briefing format |
| Streamlit threading issues | Background threads with `daemon=True` |
