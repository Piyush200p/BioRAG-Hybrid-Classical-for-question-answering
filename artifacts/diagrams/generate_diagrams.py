"""Generate editable draw.io XML plus SVG/PNG exports for BioRAG-Hybrid.

The layouts are deliberately fixed on a 16:9 canvas, with aligned process
boxes, restrained colours, and orthogonal left-to-right data flow.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
W, H = 1600, 900
PALETTE = {
    "ink": "#17324D",
    "muted": "#5A6B7B",
    "line": "#55758C",
    "data": "#DDF2F2",
    "prep": "#E8F0FB",
    "retrieval": "#FFF0CF",
    "evidence": "#EEE8FB",
    "output": "#F9E8E8",
    "eval": "#E6F4E8",
    "white": "#FFFFFF",
    "planned": "#B36B00",
}


def font(size: int, bold: bool = False):
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def wrap(text: str, width: int = 22) -> list[str]:
    explicit = text.split("\n")
    lines: list[str] = []
    for part in explicit:
        words = part.split()
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
    return lines or [""]


def svg_text(text: str, x: float, y: float, size: int, fill: str, bold: bool = False, anchor: str = "middle") -> str:
    lines = wrap(text, max(12, int(500 / size)))
    family = "DejaVu Sans"
    weight = "700" if bold else "400"
    start = y - (len(lines) - 1) * size * 0.55
    body = [f'<text x="{x}" y="{start:.1f}" fill="{fill}" font-family="{family}" font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}">']
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size * 1.25
        body.append(f'<tspan x="{x}" dy="{dy:.1f}">{html.escape(line)}</tspan>')
    body.append("</text>")
    return "".join(body)


def rounded_rect_svg(x: float, y: float, w: float, h: float, fill: str, stroke: str = PALETTE["line"], radius: int = 14, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"{dash}/>'


def cylinder_svg(x: float, y: float, w: float, h: float, fill: str, stroke: str = PALETTE["line"]) -> str:
    ry = 18
    return (
        f'<path d="M{x} {y + ry} A{w / 2} {ry} 0 0 1 {x + w} {y + ry} L{x + w} {y + h - ry} '
        f'A{w / 2} {ry} 0 0 1 {x} {y + h - ry} Z" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
        f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
    )


def arrow_svg(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{PALETTE["line"]}" stroke-width="4" marker-end="url(#arrow)"{dash}/>'


def simple_svg() -> str:
    boxes = [
        (45, 350, 190, 150, "BioASQ-QA\nquestion + corpus", PALETTE["data"]),
        (270, 350, 190, 150, "Inspect, clean\n& split by ID", PALETTE["prep"]),
        (495, 275, 190, 130, "BM25 / TF-IDF\nlexical index", PALETTE["retrieval"]),
        (495, 495, 190, 130, "Semantic proxy\nn-gram vectors", PALETTE["retrieval"]),
        (720, 385, 190, 150, "Weighted hybrid\ntop-k rank", PALETTE["retrieval"]),
        (945, 385, 190, 150, "HMM / Viterbi\nevidence states", PALETTE["evidence"]),
        (1170, 385, 190, 150, "Supported\nanswer + metrics", PALETTE["output"]),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#55758C"/></marker></defs>',
        f'<rect width="{W}" height="{H}" fill="{PALETTE["white"]}"/>',
        svg_text("BioRAG-Hybrid | Evaluation-1 workflow", 55, 90, 42, PALETTE["ink"], True, "start"),
        svg_text("Question → transparent retrieval → coherent evidence → supported output", 55, 135, 23, PALETTE["muted"], False, "start"),
        f'<line x1="55" y1="165" x2="1545" y2="165" stroke="{PALETTE["data"]}" stroke-width="10"/>',
        svg_text("DATA", 140, 280, 19, PALETTE["muted"], True),
        svg_text("PREPARATION", 365, 280, 19, PALETTE["muted"], True),
        svg_text("RETRIEVAL", 700, 215, 19, PALETTE["muted"], True),
        svg_text("EVIDENCE", 1035, 315, 19, PALETTE["muted"], True),
        svg_text("OUTPUT", 1265, 315, 19, PALETTE["muted"], True),
    ]
    for x, y, w, h, label, fill in boxes:
        parts.append(cylinder_svg(x, y, w, h, fill) if "BioASQ-QA" in label else rounded_rect_svg(x, y, w, h, fill))
        parts.append(svg_text(label.replace("\n", " "), x + w / 2, y + h / 2, 22, PALETTE["ink"], True))
    parts.extend([
        arrow_svg(243, 425, 262, 425),
        arrow_svg(468, 400, 487, 340),
        arrow_svg(468, 450, 487, 560),
        arrow_svg(693, 340, 712, 450),
        arrow_svg(693, 560, 712, 470),
        arrow_svg(918, 460, 937, 460),
        arrow_svg(1143, 460, 1162, 460),
    ])
    parts.extend([
        svg_text("Solid blocks = Evaluation-1 implementation", 55, 650, 22, PALETTE["ink"], True, "start"),
        svg_text("The later RNN/GRU summarizer consumes selected evidence and is evaluated separately.", 55, 690, 21, PALETTE["muted"], False, "start"),
        rounded_rect_svg(55, 745, 1490, 78, PALETTE["eval"], PALETTE["line"], 14),
        svg_text("Evaluation: document Recall@k / MRR  •  snippet support  •  exact answer EM/F1  •  summary ROUGE-L  •  error analysis", 800, 785, 21, PALETTE["ink"], True),
        "</svg>",
    ])
    return "".join(parts)


def group_svg(x: int, y: int, w: int, h: int, title: str, fill: str) -> list[str]:
    return [
        rounded_rect_svg(x, y, w, h, "#FBFCFD", PALETTE["line"], 18),
        svg_text(title, x + 24, y + 38, 23, PALETTE["ink"], True, "start"),
        f'<line x1="{x + 24}" y1="{y + 58}" x2="{x + w - 24}" y2="{y + 58}" stroke="{fill}" stroke-width="7"/>',
    ]


def technical_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#55758C"/></marker></defs>',
        f'<rect width="{W}" height="{H}" fill="{PALETTE["white"]}"/>',
        svg_text("BioRAG-Hybrid | technical architecture", 50, 70, 39, PALETTE["ink"], True, "start"),
        svg_text("Actual milestone components are solid; the RNN summarizer is an explicit planned extension.", 50, 110, 21, PALETTE["muted"], False, "start"),
    ]
    groups = [
        (35, 165, 330, 615, "1  DATA", PALETTE["data"]),
        (405, 165, 350, 615, "2  PREPARATION", PALETTE["prep"]),
        (795, 165, 365, 615, "3  RETRIEVAL + EVIDENCE", PALETTE["retrieval"]),
        (1200, 165, 365, 615, "4  ANSWER + EVALUATION", PALETTE["output"]),
    ]
    for group in groups:
        parts.extend(group_svg(*group))
    nodes = [
        (65, 255, 270, 90, "Question\nBioASQ body + type", PALETTE["data"], False),
        (65, 410, 270, 90, "Corpus\nabstracts + article IDs", PALETTE["data"], False),
        (65, 565, 270, 90, "Gold fields\ndocuments · snippets · answers", PALETTE["data"], False),
        (435, 245, 290, 90, "Validate schema\nmissing values + duplicates", PALETTE["prep"], False),
        (435, 385, 290, 90, "Normalize + tokenize\nkeep IDs and negation", PALETTE["prep"], False),
        (435, 525, 290, 90, "Split by question ID\n80 / 10 / 10 plan", PALETTE["prep"], False),
        (435, 665, 290, 78, "Indexes\nBM25 stats + n-gram vectors", PALETTE["prep"], False),
        (825, 245, 305, 90, "BM25 lexical score\ninterpretable anchor", PALETTE["retrieval"], False),
        (825, 385, 305, 90, "Semantic proxy score\nsubword cosine", PALETTE["retrieval"], False),
        (825, 525, 305, 90, "Weighted fusion\nrank top-k documents", PALETTE["retrieval"], False),
        (825, 665, 305, 78, "HMM / Viterbi\nSUPPORT · CONTEXT · NOISE", PALETTE["evidence"], False),
        (1230, 245, 305, 90, "Extractive answer\nselected evidence", PALETTE["output"], False),
        (1230, 385, 305, 90, "RNN / GRU summarizer\nplanned extension", "#FFF7E5", True),
        (1230, 525, 305, 90, "Metrics\nRecall@k · MRR · EM/F1", PALETTE["eval"], False),
        (1230, 665, 305, 78, "Error analysis\nsupport · negation · conflict", PALETTE["eval"], False),
    ]
    for x, y, w, h, label, fill, dashed in nodes:
        data_shape = label.startswith("Corpus") or label.startswith("Gold fields")
        parts.append(cylinder_svg(x, y, w, h, fill, PALETTE["planned"] if dashed else PALETTE["line"]) if data_shape else rounded_rect_svg(x, y, w, h, fill, PALETTE["planned"] if dashed else PALETTE["line"], 13, dashed))
        parts.append(svg_text(label.replace("\n", " "), x + w / 2, y + h / 2, 18, PALETTE["ink"], True))
    arrows = [
        (200, 345, 200, 400, False), (200, 500, 200, 555, False),
        (335, 290, 425, 290, False), (335, 455, 425, 430, False),
        (335, 610, 425, 565, False), (580, 335, 580, 380, False),
        (580, 475, 580, 520, False), (580, 615, 580, 660, False),
        (725, 705, 815, 290, False), (725, 705, 815, 430, False),
        (975, 335, 975, 515, False), (975, 475, 975, 515, False),
        (975, 615, 975, 655, False), (1130, 705, 1220, 290, False),
        (1130, 705, 1220, 430, True), (1380, 335, 1380, 515, False),
        (1380, 475, 1380, 515, True), (1380, 615, 1380, 655, False),
    ]
    for arrow in arrows:
        parts.append(arrow_svg(*arrow))
    parts.extend([
        svg_text("Dashed = planned RNN/GRU module", 50, 840, 20, PALETTE["planned"], True, "start"),
        svg_text("Every downstream metric is computed against the held-out split; gold answers never enter indexing or tuning.", 1545, 840, 18, PALETTE["muted"], False, "end"),
        "</svg>",
    ])
    return "".join(parts)


def draw_png(path: Path, technical: bool) -> None:
    image = Image.new("RGB", (W, H), PALETTE["white"])
    draw = ImageDraw.Draw(image)
    ink, muted, line = PALETTE["ink"], PALETTE["muted"], PALETTE["line"]
    draw.text((50, 30), "BioRAG-Hybrid | technical architecture" if technical else "BioRAG-Hybrid | Evaluation-1 workflow", fill=ink, font=font(39 if technical else 42, True))
    draw.text((50, 90), "Actual milestone components are solid; the RNN summarizer is an explicit planned extension." if technical else "Question → transparent retrieval → coherent evidence → supported output", fill=muted, font=font(21 if technical else 23))

    def box(x, y, w, h, label, fill, dashed=False):
        outline = PALETTE["planned"] if dashed else line
        is_cylinder = "BioASQ-QA" in label or label.startswith("Corpus") or label.startswith("Gold fields")
        if is_cylinder:
            ry = 16
            draw.rectangle((x, y + ry, x + w, y + h - ry), fill=fill, outline=outline, width=3)
            draw.ellipse((x, y, x + w, y + 2 * ry), fill=fill, outline=outline, width=3)
            draw.arc((x, y + h - 2 * ry, x + w, y + h), 0, 180, fill=outline, width=3)
        else:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill, outline=outline, width=3)
        lines = label.replace("\n", " ")
        wrapped = wrap(lines, max(12, int(w / 12)))
        total = len(wrapped) * 24
        yy = y + (h - total) / 2
        for txt in wrapped:
            tw = draw.textbbox((0, 0), txt, font=font(20 if technical else 22, True))[2]
            draw.text((x + (w - tw) / 2, yy), txt, fill=ink, font=font(20 if technical else 22, True))
            yy += 26

    def arr(x1, y1, x2, y2):
        draw.line((x1, y1, x2, y2), fill=line, width=4)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 13
        p1 = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
        p2 = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
        draw.polygon([(x2, y2), p1, p2], fill=line)

    if not technical:
        labels = [
            (45, 350, 190, 150, "BioASQ-QA\nquestion + corpus", PALETTE["data"]),
            (270, 350, 190, 150, "Inspect, clean\n& split by ID", PALETTE["prep"]),
            (495, 275, 190, 130, "BM25 / TF-IDF\nlexical index", PALETTE["retrieval"]),
            (495, 495, 190, 130, "Semantic proxy\nn-gram vectors", PALETTE["retrieval"]),
            (720, 385, 190, 150, "Weighted hybrid\ntop-k rank", PALETTE["retrieval"]),
            (945, 385, 190, 150, "HMM / Viterbi\nevidence states", PALETTE["evidence"]),
            (1170, 385, 190, 150, "Supported\nanswer + metrics", PALETTE["output"]),
        ]
        for x, y, w, h, label, fill in labels:
            box(x, y, w, h, label, fill)
        arr(243, 425, 262, 425)
        arr(468, 400, 487, 340)
        arr(468, 450, 487, 560)
        arr(693, 340, 712, 450)
        arr(693, 560, 712, 470)
        arr(918, 460, 937, 460)
        arr(1143, 460, 1162, 460)
        draw.text((55, 650), "Solid blocks = Evaluation-1 implementation", fill=ink, font=font(22, True))
        draw.text((55, 690), "The later RNN/GRU summarizer consumes selected evidence and is evaluated separately.", fill=muted, font=font(21))
        draw.rounded_rectangle((55, 745, 1545, 823), radius=14, fill=PALETTE["eval"], outline=line, width=3)
        draw.text((95, 775), "Evaluation: Recall@k / MRR  •  snippet support  •  EM/F1  •  ROUGE-L  •  error analysis", fill=ink, font=font(21, True))
    else:
        groups = [(35, 165, 330, 615, "1  DATA", PALETTE["data"]), (405, 165, 350, 615, "2  PREPARATION", PALETTE["prep"]), (795, 165, 365, 615, "3  RETRIEVAL + EVIDENCE", PALETTE["retrieval"]), (1200, 165, 365, 615, "4  ANSWER + EVALUATION", PALETTE["output"])]
        for x, y, w, h, title, color in groups:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill="#FBFCFD", outline=line, width=3)
            draw.text((x + 24, y + 22), title, fill=ink, font=font(23, True))
            draw.line((x + 24, y + 58, x + w - 24, y + 58), fill=color, width=7)
        nodes = [
            (65, 255, 270, 90, "Question\nBioASQ body + type", PALETTE["data"], False), (65, 410, 270, 90, "Corpus\nabstracts + article IDs", PALETTE["data"], False), (65, 565, 270, 90, "Gold fields\ndocuments · snippets · answers", PALETTE["data"], False),
            (435, 245, 290, 90, "Validate schema\nmissing values + duplicates", PALETTE["prep"], False), (435, 385, 290, 90, "Normalize + tokenize\nkeep IDs and negation", PALETTE["prep"], False), (435, 525, 290, 90, "Split by question ID\n80 / 10 / 10 plan", PALETTE["prep"], False), (435, 665, 290, 78, "Indexes\nBM25 stats + n-gram vectors", PALETTE["prep"], False),
            (825, 245, 305, 90, "BM25 lexical score\ninterpretable anchor", PALETTE["retrieval"], False), (825, 385, 305, 90, "Semantic proxy score\nsubword cosine", PALETTE["retrieval"], False), (825, 525, 305, 90, "Weighted fusion\nrank top-k documents", PALETTE["retrieval"], False), (825, 665, 305, 78, "HMM / Viterbi\nSUPPORT · CONTEXT · NOISE", PALETTE["evidence"], False),
            (1230, 245, 305, 90, "Extractive answer\nselected evidence", PALETTE["output"], False), (1230, 385, 305, 90, "RNN / GRU summarizer\nplanned extension", "#FFF7E5", True), (1230, 525, 305, 90, "Metrics\nRecall@k · MRR · EM/F1", PALETTE["eval"], False), (1230, 665, 305, 78, "Error analysis\nsupport · negation · conflict", PALETTE["eval"], False),
        ]
        for node in nodes:
            box(*node)
        arrs = [(200, 345, 200, 400), (200, 500, 200, 555), (335, 290, 425, 290), (335, 455, 425, 430), (335, 610, 425, 565), (580, 335, 580, 380), (580, 475, 580, 520), (580, 615, 580, 660), (725, 705, 815, 290), (725, 705, 815, 430), (975, 335, 975, 515), (975, 475, 975, 515), (975, 615, 975, 655), (1130, 705, 1220, 290), (1130, 705, 1220, 430), (1380, 335, 1380, 515), (1380, 475, 1380, 515), (1380, 615, 1380, 655)]
        for values in arrs:
            arr(*values)
        draw.text((50, 840), "Dashed = planned RNN/GRU module", fill=PALETTE["planned"], font=font(20, True))
        note = "Every downstream metric uses a held-out split; gold answers never enter indexing or tuning."
        draw.text((1545 - draw.textbbox((0, 0), note, font=font(18))[2], 840), note, fill=muted, font=font(18))
    image.save(path)


def drawio_file(path: Path, technical: bool) -> None:
    root = ET.Element("mxfile", host="app.diagrams.net", version="24.7.17")
    diagram = ET.SubElement(root, "diagram", name="Technical Architecture" if technical else "Simple Workflow")
    model = ET.SubElement(diagram, "mxGraphModel", dx="1600", dy="900", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1600", pageHeight="900", math="0", shadow="0")
    graph = ET.SubElement(model, "root")
    ET.SubElement(graph, "mxCell", id="0")
    ET.SubElement(graph, "mxCell", id="1", parent="0")

    def cell(cell_id: str, value: str, style: str, x: int, y: int, w: int, h: int, parent: str = "1"):
        node = ET.SubElement(graph, "mxCell", id=cell_id, value=html.escape(value).replace("\n", "&lt;br&gt;"), style=style, vertex="1", parent=parent)
        ET.SubElement(node, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def edge(cell_id: str, source: str, target: str, dashed: bool = False):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=3;endArrow=block;endFill=1;" + ("dashed=1;strokeColor=#B36B00;" if dashed else "strokeColor=#55758C;")
        node = ET.SubElement(graph, "mxCell", id=cell_id, value="", style=style, edge="1", parent="1", source=source, target=target)
        ET.SubElement(node, "mxGeometry", {"relative": "1", "as": "geometry"})

    base = "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;fontFamily=Helvetica;fontSize=18;fontColor=#17324D;align=center;verticalAlign=middle;"
    if not technical:
        labels = [("s1", "BioASQ-QA question + corpus", 45, 350, 190, 150, "fillColor=#DDF2F2;"), ("s2", "Inspect, clean & split by ID", 270, 350, 190, 150, "fillColor=#E8F0FB;"), ("s3", "BM25 / TF-IDF lexical index", 495, 275, 190, 130, "fillColor=#FFF0CF;"), ("s4", "Semantic proxy n-gram vectors", 495, 495, 190, 130, "fillColor=#FFF0CF;"), ("s5", "Weighted hybrid top-k rank", 720, 385, 190, 150, "fillColor=#FFF0CF;"), ("s6", "HMM / Viterbi evidence states", 945, 385, 190, 150, "fillColor=#EEE8FB;"), ("s7", "Supported answer + metrics", 1170, 385, 190, 150, "fillColor=#F9E8E8;")]
        for cid, label, x, y, w, h, color in labels:
            cell(cid, label, base + ("shape=cylinder3;" if cid == "s1" else "") + color, x, y, w, h)
        for index, (source, target) in enumerate([("s1", "s2"), ("s2", "s3"), ("s2", "s4"), ("s3", "s5"), ("s4", "s5"), ("s5", "s6"), ("s6", "s7")]):
            edge(f"se{index}", source, target)
    else:
        groups = [("g1", "1  DATA", 35, 165, 330, 615, "#DDF2F2"), ("g2", "2  PREPARATION", 405, 165, 350, 615, "#E8F0FB"), ("g3", "3  RETRIEVAL + EVIDENCE", 795, 165, 365, 615, "#FFF0CF"), ("g4", "4  ANSWER + EVALUATION", 1200, 165, 365, 615, "#F9E8E8")]
        for cid, label, x, y, w, h, color in groups:
            cell(cid, label, "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;dashed=1;fillColor=#FBFCFD;strokeColor=#55758C;fontFamily=Helvetica;fontSize=23;fontStyle=1;fontColor=#17324D;verticalAlign=top;align=left;spacingTop=12;spacingLeft=20;", x, y, w, h)
        nodes = [("t1", "Question | BioASQ body + type", 65, 255, 270, 90, "#DDF2F2"), ("t2", "Corpus | abstracts + article IDs", 65, 410, 270, 90, "#DDF2F2"), ("t3", "Gold fields | documents, snippets, answers", 65, 565, 270, 90, "#DDF2F2"), ("t4", "Validate schema | missing values + duplicates", 435, 245, 290, 90, "#E8F0FB"), ("t5", "Normalize + tokenize | keep IDs and negation", 435, 385, 290, 90, "#E8F0FB"), ("t6", "Split by question ID | 80 / 10 / 10 plan", 435, 525, 290, 90, "#E8F0FB"), ("t7", "Indexes | BM25 stats + n-gram vectors", 435, 665, 290, 78, "#E8F0FB"), ("t8", "BM25 lexical score | interpretable anchor", 825, 245, 305, 90, "#FFF0CF"), ("t9", "Semantic proxy score | subword cosine", 825, 385, 305, 90, "#FFF0CF"), ("t10", "Weighted fusion | rank top-k documents", 825, 525, 305, 90, "#FFF0CF"), ("t11", "HMM / Viterbi | SUPPORT, CONTEXT, NOISE", 825, 665, 305, 78, "#EEE8FB"), ("t12", "Extractive answer | selected evidence", 1230, 245, 305, 90, "#F9E8E8"), ("t13", "RNN / GRU summarizer | planned extension", 1230, 385, 305, 90, "#FFF7E5"), ("t14", "Metrics | Recall@k, MRR, EM/F1", 1230, 525, 305, 90, "#E6F4E8"), ("t15", "Error analysis | support, negation, conflict", 1230, 665, 305, 78, "#E6F4E8")]
        for cid, label, x, y, w, h, color in nodes:
            shape = "shape=cylinder3;" if cid in {"t2", "t3"} else ""
            style = base + shape + f"fillColor={color};" + ("dashed=1;strokeColor=#B36B00;" if cid == "t13" else "strokeColor=#55758C;")
            cell(cid, label, style, x, y, w, h)
        for a, b in [("t1", "t2"), ("t2", "t3"), ("t1", "t4"), ("t2", "t5"), ("t3", "t6"), ("t4", "t5"), ("t5", "t6"), ("t6", "t7"), ("t7", "t8"), ("t7", "t9"), ("t8", "t10"), ("t9", "t10"), ("t10", "t11"), ("t11", "t12"), ("t11", "t13"), ("t12", "t14"), ("t13", "t14"), ("t14", "t15")]:
            edge("e" + a + b, a, b, a == "t11" and b == "t13" or a == "t13")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    outputs = [
        ("simple_workflow", False, simple_svg()),
        ("technical_architecture", True, technical_svg()),
    ]
    for stem, technical, svg in outputs:
        (ROOT / f"{stem}.svg").write_text(svg, encoding="utf-8")
        draw_png(ROOT / f"{stem}.png", technical)
        drawio_file(ROOT / f"{stem}.drawio", technical)
    (ROOT / "simple_workflow.mmd").write_text("""flowchart LR\n  A[(BioASQ-QA JSON)] --> B[Inspect + validate]\n  B --> C[Normalize + split by question ID]\n  C --> D[BM25 / TF-IDF]\n  C --> E[Distributional n-gram representation]\n  D --> F[Weighted hybrid rank]\n  E --> F\n  F --> G[Candidate documents + sentences]\n  G --> H[HMM / Viterbi evidence sequence]\n  H --> I[Extractive supported answer]\n  I --> J[Recall@k · MRR · F1 · ROUGE-L]\n  J --> K[Error + support analysis]\n""", encoding="utf-8")
    (ROOT / "technical_architecture.mmd").write_text("""flowchart TB\n  Q[BioASQ question + type] --> P[Validate + normalize + split]\n  C[(Abstract corpus)] --> P\n  P --> L[BM25 lexical index]\n  P --> S[Distributional n-gram vectors]\n  L --> F[Weighted hybrid retrieval]\n  S --> F\n  F --> H[HMM/Viterbi evidence selector]\n  H --> X[Extractive supported answer]\n  H -. planned .-> R[RNN/GRU summarizer]\n  X --> E[Recall@k, MRR, F1, ROUGE-L, support review]\n  R --> E\n""", encoding="utf-8")
    print(f"Generated {len(outputs)} diagram sets in {ROOT}")


if __name__ == "__main__":
    main()
