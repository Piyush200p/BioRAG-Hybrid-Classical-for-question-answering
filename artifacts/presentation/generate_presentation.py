"""Build the editable BioRAG-Hybrid Evaluation-1 PowerPoint."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "presentation" / "BioRAG_Hybrid_Evaluation1.pptx"
DIAGRAMS = ROOT / "artifacts" / "diagrams"

NAVY = RGBColor(23, 50, 77)
MUTED = RGBColor(90, 107, 123)
TEAL = RGBColor(33, 151, 161)
PALE_TEAL = RGBColor(221, 242, 242)
PALE_BLUE = RGBColor(232, 240, 251)
PALE_AMBER = RGBColor(255, 240, 207)
PALE_PURPLE = RGBColor(238, 232, 251)
PALE_GREEN = RGBColor(230, 244, 232)
PALE_PINK = RGBColor(249, 232, 232)
AMBER = RGBColor(179, 107, 0)
WHITE = RGBColor(255, 255, 255)
LIGHT = RGBColor(247, 250, 252)


def set_bg(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text_box(slide, text, x, y, w, h, size=18, color=NAVY, bold=False, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, font="Aptos", margin=0.04):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    paragraph = tf.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def rich_text(slide, lines, x, y, w, h, size=18, color=NAVY, bullet=True, gap=4):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    for index, item in enumerate(lines):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.space_after = Pt(gap)
        p.alignment = PP_ALIGN.LEFT
        if bullet:
            p.text = "• " + item
        for run in p.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(size)
            run.font.color.rgb = color
    return shape


def rect(slide, x, y, w, h, fill=LIGHT, line=RGBColor(206, 220, 230), radius=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.3)
    return shape


def header(slide, title, kicker=None, page=None):
    if kicker:
        text_box(slide, kicker.upper(), 0.55, 0.22, 4.5, 0.28, 10, TEAL, True)
    text_box(slide, title, 0.55, 0.50, 12.1, 0.55, 27, NAVY, True)
    slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.16), Inches(12.25), Inches(0.035)).fill.solid()
    bar = slide.shapes[-1]
    bar.fill.fore_color.rgb = PALE_TEAL
    bar.line.fill.background()
    if page is not None:
        text_box(slide, f"{page:02d}", 12.35, 0.25, 0.45, 0.25, 10, MUTED, True, PP_ALIGN.RIGHT)


def footer(slide, source="Team 4 • BioRAG-Hybrid • Project Evaluation 1"):
    text_box(slide, source, 0.55, 7.16, 11.7, 0.18, 7.5, MUTED)


def card(slide, title, body, x, y, w, h, fill=LIGHT, title_color=TEAL, body_size=15):
    rect(slide, x, y, w, h, fill)
    text_box(slide, title, x + 0.15, y + 0.14, w - 0.3, 0.35, 16, title_color, True)
    text_box(slide, body, x + 0.15, y + 0.56, w - 0.3, h - 0.68, body_size, NAVY)


def add_table(slide, data, x, y, w, h, font_size=10, header_fill=NAVY):
    rows, cols = len(data), len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    for col in range(cols):
        table.columns[col].width = Inches(w / cols)
    for row in range(rows):
        for col in range(cols):
            cell = table.cell(row, col)
            cell.text = str(data[row][col])
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if row == 0 else (WHITE if row % 2 else LIGHT)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT
                for run in paragraph.runs:
                    run.font.name = "Aptos"
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = WHITE if row == 0 else NAVY
                    run.font.bold = row == 0
    return table_shape


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1
    slide = prs.slides.add_slide(blank); set_bg(slide, NAVY)
    text_box(slide, "BIO RAG\nHYBRID", 0.75, 1.25, 5.4, 1.7, 42, WHITE, True)
    text_box(slide, "Classical–neural biomedical retrieval-augmented question answering", 0.8, 3.2, 6.2, 0.7, 23, PALE_TEAL)
    rect(slide, 8.1, 1.3, 4.1, 3.9, PALE_TEAL, PALE_TEAL)
    text_box(slide, "TEAM 4", 8.55, 1.85, 3.2, 0.45, 19, TEAL, True, PP_ALIGN.CENTER)
    text_box(slide, "PROJECT\nEVALUATION 1", 8.55, 2.55, 3.2, 1.0, 26, NAVY, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text_box(slide, "Student names / roll numbers\n[insert from team list]", 8.55, 4.1, 3.2, 0.6, 13, MUTED, False, PP_ALIGN.CENTER)
    text_box(slide, "Literature • evidence • retrieval • reproducibility", 0.8, 6.55, 7.0, 0.3, 13, PALE_TEAL)

    # 2
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Problem statement", "01 • motivation", 2)
    card(slide, "Biomedical questions need both recall and support", "A relevant-looking passage can still be insufficient, outdated, or contradictory. The system must expose the path from question to evidence to answer.", 0.7, 1.65, 5.75, 2.25, PALE_BLUE, TEAL, 18)
    card(slide, "Research question", "Does hybrid lexical–semantic retrieval plus HMM/Viterbi evidence selection improve supported answering over single-signal baselines on BioASQ Task B-style data?", 6.85, 1.65, 5.75, 2.25, PALE_AMBER, AMBER, 18)
    text_box(slide, "Question", 1.05, 4.7, 1.3, 0.35, 16, NAVY, True, PP_ALIGN.CENTER)
    text_box(slide, "Documents", 3.05, 4.7, 1.5, 0.35, 16, NAVY, True, PP_ALIGN.CENTER)
    text_box(slide, "Evidence", 5.3, 4.7, 1.3, 0.35, 16, NAVY, True, PP_ALIGN.CENTER)
    text_box(slide, "Supported answer", 7.65, 4.7, 2.0, 0.35, 16, NAVY, True, PP_ALIGN.CENTER)
    text_box(slide, "Inspectability is the design principle", 4.25, 5.75, 4.8, 0.45, 22, TEAL, True, PP_ALIGN.CENTER)
    footer(slide, "Source: uploaded proposal images; Krithara et al. (2023)")

    # 3
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Motivation and applications", "02 • why this matters", 3)
    card(slide, "Literature scale", "Biomedical records are large, terminology-heavy, and continuously updated. Manual search must be accelerated without hiding the supporting evidence.", 0.7, 1.7, 3.8, 2.4, PALE_TEAL, TEAL, 16)
    card(slide, "Language difficulty", "Synonyms, abbreviations, spelling variants, long abstracts, and negated claims challenge keyword-only matching.", 4.78, 1.7, 3.8, 2.4, PALE_AMBER, AMBER, 16)
    card(slide, "Applications", "Literature review, biomedical information access, research-question triage, and evidence organization. Not a clinical diagnosis tool.", 8.86, 1.7, 3.8, 2.4, PALE_PURPLE, TEAL, 16)
    text_box(slide, "Design goal: make every retrieval and evidence decision inspectable.", 1.0, 5.35, 11.3, 0.6, 24, NAVY, True, PP_ALIGN.CENTER)
    footer(slide, "Source: uploaded proposal images; BioASQ-QA descriptor")

    # 4
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Objectives and scope", "03 • project definition", 4)
    rich_text(slide, ["Load and validate BioASQ-shaped JSON.", "Compare BM25, semantic proxy, and weighted hybrid rankers.", "Select coherent evidence with HMM/Viterbi.", "Produce short answers plus supporting snippets; add RNN/GRU after the baseline.", "Evaluate retrieval, answer, summary, and support quality."], 0.85, 1.65, 6.2, 3.8, 19)
    rect(slide, 7.55, 1.65, 4.65, 3.7, PALE_BLUE)
    text_box(slide, "Scope boundary", 7.9, 2.0, 3.9, 0.4, 20, TEAL, True, PP_ALIGN.CENTER)
    text_box(slide, "English BioASQ Task B-style literature QA\n\nNo EHR data\nNo diagnosis or treatment recommendation\nNo unsupported state-of-the-art claim", 7.95, 2.75, 3.85, 1.9, 18, NAVY, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    footer(slide, "Design choice: narrow the initial experiment so it is measurable and reproducible")

    # 5
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Existing approaches and shortcomings", "04 • research context", 5)
    data = [["Approach", "Strength", "Limitation motivating us"], ["BM25 / TF-IDF", "Fast, transparent", "Misses paraphrase and synonym links"], ["Dense retrieval", "Semantic matching", "May retrieve related but unsupported text"], ["Vanilla RAG", "Adds external context", "Context noise and corpus sensitivity"], ["Learned filtering", "Estimates evidence utility", "Extra training/compute; less transparent"]]
    add_table(slide, data, 0.7, 1.7, 11.95, 3.3, 13)
    text_box(slide, "Gap direction: combine complementary signals, then expose a sentence-level evidence decision.", 1.05, 5.6, 11.2, 0.55, 23, TEAL, True, PP_ALIGN.CENTER)
    footer(slide, "Citations: Xiong et al. (2024); Sohn et al. (2025)")

    # 6
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Paper 1 — MedRAG / MIRAGE", "05 • literature review", 6)
    text_box(slide, "Xiong, Jin, Lu & Zhang • Findings ACL 2024", 0.7, 1.45, 7.5, 0.3, 14, TEAL, True)
    card(slide, "What it does", "MIRAGE: 7,663 questions from five medical QA datasets. MedRAG compares PubMed, StatPearls, textbooks and Wikipedia with BM25, Contriever, SPECTER, MedCPT, RRF and six LLMs.", 0.7, 1.95, 5.75, 2.1, PALE_BLUE, TEAL, 15)
    card(slide, "Findings", "Up to 18% improvement over CoT. GPT-3.5 with MedCorp + RRF-4: 71.57 average accuracy. Corpus choice and context size matter.", 6.85, 1.95, 5.75, 2.1, PALE_GREEN, TEAL, 15)
    card(slide, "Limitations → our lesson", "Vanilla RAG, high compute, multi-choice emphasis, limited retrieval labels, and no sentence-coherence layer. We adapt the BM25/semantic ablation idea.", 0.7, 4.45, 11.9, 1.35, PALE_AMBER, AMBER, 15)
    footer(slide, "Official: https://aclanthology.org/2024.findings-acl.372/ • DOI 10.18653/v1/2024.findings-acl.372")

    # 7
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Paper 2 — RAG²", "06 • literature review", 7)
    text_box(slide, "Sohn et al. • NAACL 2025 long paper", 0.7, 1.45, 7.5, 0.3, 14, TEAL, True)
    stages = [("Rationale query", PALE_TEAL), ("Balanced retrieval", PALE_AMBER), ("MedCPT rerank", PALE_PURPLE), ("Flan-T5 filter", PALE_PINK), ("Answer", PALE_GREEN)]
    x = 0.8
    for title, fill in stages:
        rect(slide, x, 2.0, 2.25, 0.7, fill)
        text_box(slide, title, x + 0.1, 2.2, 2.05, 0.25, 13, NAVY, True, PP_ALIGN.CENTER)
        x += 2.4
    card(slide, "Results", "RAG² average accuracy: 66.3 on Llama-3-8B-Instruct, 72.4 on Meerkat-7B, 86.9 on GPT-4o across MedQA, MedMCQA and MMLU-Med.", 0.7, 3.25, 5.75, 1.9, PALE_GREEN, TEAL, 15)
    card(slide, "Limitations → our lesson", "Rationale errors can distract retrieval; filters operate per snippet; perplexity is a proxy; the method adds cost and is not focused on BioASQ paragraph summaries. We adapt the failure-analysis mindset, not the full compute footprint.", 6.85, 3.25, 5.75, 1.9, PALE_AMBER, AMBER, 15)
    footer(slide, "Official: https://aclanthology.org/2025.naacl-long.635/ • DOI 10.18653/v1/2025.naacl-long.635")

    # 8
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Paper 3 — BioASQ-QA", "07 • dataset paper", 8)
    text_box(slide, "Krithara, Nentidis, Bougiatiotis & Paliouras • Scientific Data 2023", 0.7, 1.45, 9.6, 0.3, 14, TEAL, True)
    card(slide, "Data model", "English expert questions with relevant articles, snippets, exact answers for short-answer types, and ideal paragraph answers. Types: yes/no, factoid, list, summary.", 0.7, 1.95, 5.75, 2.2, PALE_TEAL, TEAL, 16)
    card(slide, "Scale and validation", "4,721 questions reported at publication; inspected BioASQ11 file contains 4,719 records. Expert evaluation covers information recall, precision, repetition and readability.", 6.85, 1.95, 5.75, 2.2, PALE_BLUE, TEAL, 16)
    card(slide, "Limitation → our lesson", "Document/snippet evidence can disagree across experts even when ideal answers agree. We report retrieval and summary quality separately and never treat one evidence path as absolute truth.", 0.7, 4.65, 11.9, 1.2, PALE_AMBER, AMBER, 15)
    footer(slide, "Official: https://doi.org/10.1038/s41597-023-02068-4 • Zenodo: 10.5281/zenodo.7655130")

    # 9
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Comparison and research gap", "08 • synthesis", 9)
    data = [["Paper", "What we learn", "What remains for us"], ["MedRAG / MIRAGE", "Corpus/retriever choices matter", "Test the same principle on evidence + summaries"], ["RAG²", "Similarity alone is not evidence utility", "Use an inspectable sequence selector"], ["BioASQ-QA", "Evidence and ideal answers coexist", "Run a leakage-safe BioASQ ablation"]]
    add_table(slide, data, 0.7, 1.55, 11.95, 2.65, 12)
    rect(slide, 1.2, 4.85, 10.9, 1.05, PALE_PURPLE)
    text_box(slide, "Inferred gap: a low-cost, transparent BioASQ experiment comparing lexical/semantic fusion plus sequence-level evidence selection.", 1.5, 5.08, 10.3, 0.55, 20, NAVY, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text_box(slide, "Hypothesis — not yet verified: hybrid ranking + HMM can improve evidence coverage/coherence.", 1.1, 6.2, 11.1, 0.35, 14, AMBER, True, PP_ALIGN.CENTER)
    footer(slide, "The project claims a testable contribution, not state-of-the-art novelty")

    # 10
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Proposed methodology", "09 • system design", 10)
    steps = [("1", "Normalize + split", PALE_BLUE), ("2", "BM25 lexical", PALE_AMBER), ("3", "Semantic proxy", PALE_AMBER), ("4", "Weighted fusion", PALE_AMBER), ("5", "HMM / Viterbi", PALE_PURPLE), ("6", "Extractive answer", PALE_PINK)]
    x = 0.55
    for number, label, fill in steps:
        rect(slide, x, 2.05, 1.9, 1.15, fill)
        text_box(slide, number, x + 0.1, 2.15, 0.35, 0.35, 17, TEAL, True, PP_ALIGN.CENTER)
        text_box(slide, label, x + 0.15, 2.55, 1.6, 0.4, 15, NAVY, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        if x < 10.3:
            text_box(slide, "→", x + 1.93, 2.42, 0.35, 0.35, 22, MUTED, True, PP_ALIGN.CENTER)
        x += 2.12
    card(slide, "Why this order?", "Evidence is isolated before generation. The first milestone can run on CPU and lets us attribute errors to retrieval, selection, or answer formatting. RNN/GRU generation is added only after the split and evidence labels are stable.", 1.0, 4.35, 11.3, 1.45, PALE_TEAL, TEAL, 17)
    footer(slide, "Solid components are implemented in the workspace; RNN/GRU is planned")

    # 11
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "End-to-end workflow", "10 • data flow", 11)
    slide.shapes.add_picture(str(DIAGRAMS / "simple_workflow.png"), Inches(0.55), Inches(1.35), width=Inches(12.25), height=Inches(6.0))
    footer(slide, "Editable source: simple_workflow.drawio / simple_workflow.mmd")

    # 12
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Dataset — BioASQ-QA / Task B", "11 • primary dataset", 12)
    card(slide, "Source and volume", "BioASQ portal + Zenodo 10.5281/zenodo.7655130\nOfficially reported: 4,721 questions at the 2023 publication date. Inspected BioASQ11: 4,719 records.", 0.7, 1.5, 5.75, 1.55, PALE_TEAL, TEAL, 15)
    card(slide, "Format and targets", "English biomedical JSON; documents, snippets, exact answers, ideal paragraph answers; yes/no, factoid, list and summary questions.", 6.85, 1.5, 5.75, 1.55, PALE_BLUE, TEAL, 15)
    card(slide, "Split and access", "Proposed local split: 80/10/10 by question ID from released training data. Official test only when authorized. BioASQ11: open access, CC BY 2.5. JSON has URLs/snippets, not article abstracts.", 0.7, 3.35, 5.75, 1.55, PALE_AMBER, AMBER, 15)
    card(slide, "Why it fits", "One resource supports retrieval, evidence selection, short answers and paragraph summarization—the complete project story in the uploaded images.", 6.85, 3.35, 5.75, 1.55, PALE_PURPLE, TEAL, 15)
    text_box(slide, "Official URL: https://participants-area.bioasq.org/datasets/", 0.9, 5.7, 11.5, 0.35, 16, NAVY, True, PP_ALIGN.CENTER)
    footer(slide, "Source: Krithara et al. (2023); BioASQ official data portal")

    # 13
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "System architecture, tools and interfaces", "12 • implementation design", 13)
    slide.shapes.add_picture(str(DIAGRAMS / "technical_architecture.png"), Inches(0.55), Inches(1.25), width=Inches(12.25), height=Inches(6.0))
    footer(slide, "Editable source: technical_architecture.drawio / technical_architecture.mmd")

    # 14
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Implementation completed so far", "13 • genuine evidence", 14)
    card(slide, "Executed modules", "BioASQ-shaped loader\nNormalization and tokenization\nBM25 + n-gram hybrid retrieval\nThree-state HMM/Viterbi selector\nExtractive answer + metrics\nUnit tests and reproducible CLI", 0.7, 1.55, 5.7, 3.85, PALE_TEAL, TEAL, 16)
    card(slide, "Observed run", "Fixture: 6 synthetic BioASQ-shaped records\n\nRecall@3  1.000000\nMRR  1.000000\nSummary F1  0.443306\nROUGE-L F1  0.429417\n\n11/11 unit tests passed", 6.9, 1.55, 5.7, 3.85, PALE_GREEN, TEAL, 16)
    text_box(slide, "Important: these are smoke-fixture metrics, not official BioASQ results.", 1.0, 5.95, 11.3, 0.38, 16, AMBER, True, PP_ALIGN.CENTER)
    footer(slide, "Evidence: artifacts/results/demo_results.json • tests/test_pipeline.py")

    # 15
    slide = prs.slides.add_slide(blank); set_bg(slide); header(slide, "Remaining plan, risks and fallback", "14 • execution plan", 15)
    data = [["Next module", "Checkpoint", "Risk / fallback"], ["Article-text corpus", "Versioned corpus aligned to BioASQ URLs", "Access/version drift → retain fixture pipeline"], ["Ablation", "BM25 vs semantic vs hybrid; top sentence vs HMM", "Evidence disagreement → report per type + manual sample"], ["Neural encoder", "Pinned environment + licence log", "GPU/storage → compact encoder or CPU proxy"], ["RNN/GRU summary", "Train/validation curves + ROUGE-L", "Overfit/weak evidence → retain extractive answer"]]
    add_table(slide, data, 0.65, 1.55, 12.05, 3.55, 11)
    text_box(slide, "The safest fallback is a well-evaluated extractive evidence system, not an unverified neural demo.", 1.0, 5.8, 11.3, 0.45, 19, TEAL, True, PP_ALIGN.CENTER)
    footer(slide, "Milestone detail: docs/08_milestone_checklist.md")

    # 16
    slide = prs.slides.add_slide(blank); set_bg(slide, NAVY)
    text_box(slide, "Evaluation criteria and references", 0.65, 0.55, 8.4, 0.6, 28, WHITE, True)
    text_box(slide, "What we will measure", 0.75, 1.55, 3.4, 0.35, 18, PALE_TEAL, True)
    rich_text(slide, ["Document Recall@k and MRR", "Snippet support and evidence precision", "Exact answer EM/F1 by question type", "ROUGE-L plus manual support review", "Negation and conflict error analysis"], 0.75, 2.0, 4.8, 2.4, 16, WHITE)
    text_box(slide, "Selected references", 6.3, 1.55, 3.5, 0.35, 18, PALE_TEAL, True)
    text_box(slide, "Xiong et al. (2024), Findings ACL\nSohn et al. (2025), NAACL\nKrithara et al. (2023), Scientific Data\n\nFull citations and links:\ndocs/02_literature_review.md\ndocs/references.bib", 6.3, 2.0, 5.5, 2.7, 15, WHITE)
    rect(slide, 0.75, 5.45, 11.8, 0.8, PALE_TEAL, PALE_TEAL)
    text_box(slide, "Contribution claim: a testable, inspectable pipeline—not an unsupported accuracy claim.", 1.0, 5.68, 11.3, 0.3, 19, NAVY, True, PP_ALIGN.CENTER)
    text_box(slide, "Thank you • Questions", 0.75, 6.65, 4.0, 0.35, 18, WHITE, True)

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
