"""
output/pdf_export.py
────────────────────
PDF export for all 4 output formats.
Owned by Devang.

Converts the final output into a downloadable PDF file.
Returns bytes that Streamlit can serve via st.download_button().
"""

from fpdf import FPDF
import re
from core.context import SharedContext


class LiteraturePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Literature Analysis System", align="R")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _clean_text(text: str) -> str:
    """Remove markdown symbols for plain PDF text."""
    text = re.sub(r"#{1,6}\s*", "", text)       # headers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text) # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # italic
    text = re.sub(r"`(.*?)`", r"\1", text)        # code
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text) # links
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE) # bullets
    text = re.sub(r"\|.*?\|", "", text)           # tables
    text = re.sub(r"-{3,}", "", text)             # hr lines
    return text.strip()


def generate_pdf(context: SharedContext) -> bytes:
    """
    Generate a PDF of the final output.

    Args:
        context: SharedContext with final_output populated.

    Returns:
        PDF as bytes — pass directly to st.download_button().
    """
    pdf = LiteraturePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title page ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(26, 26, 46)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Literature Analysis", align="C")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(0, 8, context.topic, align="C")
    pdf.ln(6)

    # Meta info
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Papers analyzed: {len(context.user_confirmed_papers)}", align="C")
    pdf.ln(6)
    pdf.cell(0, 6, f"Output format: {context.output_format.replace('_', ' ').title()}", align="C")
    pdf.ln(6)
    pdf.cell(0, 6, f"Contradictions found: {len(context.contradiction_report)}", align="C")
    pdf.ln(6)
    pdf.cell(0, 6, f"Paradigm shifts: {len(context.paradigm_timeline)}", align="C")
    pdf.ln(12)

    # Divider
    pdf.set_draw_color(76, 155, 232)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # ── Main content ──────────────────────────────────────────────────────────
    output = context.final_output

    if isinstance(output, str):
        # Written review or briefing — render markdown as plain text
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue

            # Detect heading levels
            if line.startswith("### "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(26, 26, 46)
                pdf.multi_cell(0, 7, _clean_text(line))
                pdf.ln(1)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(21, 101, 192)
                pdf.multi_cell(0, 8, _clean_text(line))
                pdf.ln(2)
            elif line.startswith("# "):
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(26, 26, 46)
                pdf.multi_cell(0, 9, _clean_text(line))
                pdf.ln(3)
            elif line.startswith("---"):
                pdf.set_draw_color(200, 200, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(4)
            else:
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                cleaned = _clean_text(line)
                if cleaned:
                    pdf.multi_cell(0, 6, cleaned)

    else:
        # For timeline/knowledge_map — export the data as structured text
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(21, 101, 192)
        pdf.cell(0, 8, "Analysis Results Summary")
        pdf.ln(10)

        # Papers section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 7, "Top Papers")
        pdf.ln(8)

        for i, paper in enumerate(context.user_confirmed_papers[:10], 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 50, 50)
            title = paper.get("title", "Untitled")[:80]
            pdf.multi_cell(0, 6, f"{i}. {title}")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(120, 120, 120)
            authors = paper.get("authors", [])
            author_str = ", ".join(authors[:2])
            pdf.cell(0, 5, f"   {paper.get('year', 'N/A')} · {author_str} · {paper.get('citationCount', 0):,} citations")
            pdf.ln(6)

        # Paradigm timeline
        if context.paradigm_timeline:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(26, 26, 46)
            pdf.cell(0, 7, "Paradigm Timeline")
            pdf.ln(8)

            for event in context.paradigm_timeline:
                impact = event.get("impact", "incremental")
                icon = {"paradigm-shift": "[!!!]", "significant": "[!!]", "incremental": "[!]"}.get(impact, "[ ]")
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(0, 6, f"{event.get('year', '?')} {icon}")
                pdf.ln(5)
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(0, 5, event.get("shift_description", ""))
                pdf.ln(4)

        # Contradictions
        if context.contradiction_report:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(26, 26, 46)
            pdf.cell(0, 7, "Contradictions Found")
            pdf.ln(8)

            for c in context.contradiction_report[:5]:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(200, 50, 50)
                pdf.cell(0, 5, f"[{c.get('severity', 'minor').upper()}]")
                pdf.ln(5)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 5, f"{c.get('paper_a_title', '?')[:50]} vs {c.get('paper_b_title', '?')[:50]}")
                pdf.multi_cell(0, 5, c.get("reasoning", "")[:150])
                pdf.ln(4)

    return bytes(pdf.output())
