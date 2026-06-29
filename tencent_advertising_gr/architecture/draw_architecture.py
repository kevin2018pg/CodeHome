#!/usr/bin/env python3
"""Draw the tencent_advertising_gr architecture as a paper-style SVG."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "tencent_advertising_gr_architecture.svg"


W, H = 2200, 1350


def text(x, y, value, size=28, weight="400", anchor="middle", color="#111827"):
    value = escape(value)
    return (
        f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{value}</text>'
    )


def multiline(x, y, lines, size=24, weight="400", anchor="middle", color="#111827", leading=1.28):
    parts = [
        f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * leading
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def rect(x, y, w, h, label, sub=None, fill="#ffffff", stroke="#111827", sw=2.2, r=10):
    s = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    if isinstance(label, str):
        label_lines = [label]
    else:
        label_lines = label
    s += multiline(x + w / 2, y + 34, label_lines, size=25, weight="700")
    if sub:
        s += multiline(x + w / 2, y + 74, sub, size=19, color="#374151")
    return s


def group_box(x, y, w, h, title, fill="#f8fafc", stroke="#94a3b8"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2.4" stroke-dasharray="8 7"/>'
        + text(x + 22, y + 38, title, size=26, weight="700", anchor="start", color="#0f172a")
    )


def arrow(x1, y1, x2, y2, color="#111827", sw=2.6, label=None, label_pos=0.5):
    s = (
        f'<path d="M{x1},{y1} C{(x1+x2)/2},{y1} {(x1+x2)/2},{y2} {x2},{y2}" '
        f'fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arrow)"/>'
    )
    if label:
        lx = x1 + (x2 - x1) * label_pos
        ly = y1 + (y2 - y1) * label_pos - 10
        s += text(lx, ly, label, size=18, weight="600", color=color)
    return s


def straight_arrow(x1, y1, x2, y2, color="#111827", sw=2.6, label=None):
    s = (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" marker-end="url(#arrow)"/>'
    )
    if label:
        s += text((x1 + x2) / 2, (y1 + y2) / 2 - 10, label, size=18, weight="600", color=color)
    return s


def main():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1350" viewBox="0 0 2200 1350">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#111827"/>',
        "</marker>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.12"/>',
        "</filter>",
        "</defs>",
        '<rect width="2200" height="1350" fill="#ffffff"/>',
        text(1100, 58, "tencent_advertising_gr: Generative Sequential Recommendation Architecture", 36, "800"),
        text(1100, 94, "HSTU-based action-conditioned next-item retrieval with mixed negative sampled softmax", 22, "500", color="#475569"),
    ]

    # Group boxes
    parts += [
        group_box(40, 140, 430, 1030, "1. Data & Sampling"),
        group_box(520, 140, 455, 1030, "2. Feature Encoding"),
        group_box(1025, 140, 565, 1030, "3. HSTU Sequence Model"),
        group_box(1640, 140, 500, 500, "4. Training Objective"),
        group_box(1640, 700, 500, 470, "5. Inference Retrieval"),
    ]

    # Data and sampling
    parts += [
        rect(80, 205, 350, 130, "Raw Logs", ["seq.jsonl: user, item, features", "action type, timestamp"], "#f8fafc"),
        rect(80, 370, 350, 130, "Metadata", ["indexer.pkl, item_feat_dict", "optional creative_emb memmap"], "#f8fafc"),
        rect(80, 535, 350, 130, "Statistics", ["get_stat.py", "item_freq.csv, item_last_ts.json"], "#f8fafc"),
        rect(80, 715, 350, 150, "Dataset Collate", ["left-padded sequences", "token_type: item=1, user=2", "packed feature tensors"], "#eef2ff", "#4338ca"),
        rect(80, 920, 350, 150, "Targets & Negatives", ["positive next item", "global active negatives", "exclude user-seen items"], "#eef2ff", "#4338ca"),
    ]

    # Feature encoding
    parts += [
        rect(560, 230, 375, 130, "User Tower", ["user_id embedding", "user sparse / array features"], "#f0fdf4", "#15803d"),
        rect(560, 430, 375, 160, "Item Tower", ["item_id + sparse / array", "continuous + dense multimodal", "MLP -> 512-d item vector"], "#f0fdf4", "#15803d"),
        rect(560, 660, 375, 140, "FiLM Conditioning", ["aggregate user vector u", "gamma, beta = Linear(u)", "x = x * (1 + gamma) + beta"], "#ecfdf5", "#047857"),
        rect(560, 890, 375, 160, "Context Injection", ["cyclic time sin/cos", "next-action embedding", "position embedding"], "#ecfdf5", "#047857"),
    ]

    # HSTU
    parts += [
        rect(1070, 215, 475, 105, "Sequence Input", ["conditioned token embeddings x_t"], "#fff7ed", "#c2410c"),
        rect(1070, 365, 475, 120, "UVQK Projection", ["single linear -> U, V, Q, K", "multi-head split"], "#fff7ed", "#c2410c"),
        rect(1070, 530, 475, 145, "Pointwise Attention", ["A = SiLU(QK^T / sqrt(d) + RAB)", "causal + padding mask", "no softmax"], "#fffbeb", "#b45309"),
        rect(1070, 725, 475, 120, "Temporal RAB", ["log-bucketed timestamp difference", "shared across HSTU layers"], "#fefce8", "#a16207"),
        rect(1070, 890, 475, 140, "Gated Transduction", ["Y = A @ V", "LayerNorm(Y) * U", "output projection + residual"], "#fff7ed", "#c2410c"),
    ]

    # Training objective
    parts += [
        rect(1685, 220, 410, 120, "Query & Positives", ["Q: HSTU output per token", "K+: encoded next item"], "#fdf2f8", "#be185d"),
        rect(1685, 385, 410, 125, "Mixed Negatives", ["in-batch positives", "global random negatives", "false-negative filtering"], "#fdf2f8", "#be185d"),
        rect(1685, 545, 410, 70, "Loss: sampled softmax with -log q(i)", ["hard top-k curriculum"], "#fce7f3", "#9d174d"),
    ]

    # Inference
    parts += [
        rect(1685, 770, 410, 115, "Candidate Encoding", ["predict_set.jsonl", "item tower -> normalized matrix"], "#eff6ff", "#1d4ed8"),
        rect(1685, 930, 410, 115, "User Query", ["predict_seq.jsonl history", "target action condition", "final HSTU state"], "#eff6ff", "#1d4ed8"),
        rect(1685, 1080, 410, 65, "Top-K Retrieval", ["chunked dot product, mask seen items"], "#dbeafe", "#1e40af"),
    ]

    # Arrows
    parts += [
        straight_arrow(255, 335, 255, 370),
        straight_arrow(255, 500, 255, 535),
        straight_arrow(255, 665, 255, 715),
        straight_arrow(255, 865, 255, 920),
        arrow(430, 790, 560, 495),
        arrow(430, 790, 560, 295),
        straight_arrow(748, 360, 748, 660),
        straight_arrow(748, 590, 748, 660),
        straight_arrow(748, 800, 748, 890),
        arrow(935, 970, 1070, 270),
        straight_arrow(1308, 320, 1308, 365),
        straight_arrow(1308, 485, 1308, 530),
        arrow(1308, 725, 1308, 675),
        straight_arrow(1308, 675, 1308, 890),
        arrow(1545, 960, 1685, 280),
        arrow(430, 995, 1685, 280),
        arrow(430, 995, 1685, 445),
        straight_arrow(1890, 340, 1890, 385),
        straight_arrow(1890, 510, 1890, 545),
        arrow(1545, 960, 1685, 985),
        straight_arrow(1890, 885, 1890, 930),
        straight_arrow(1890, 1045, 1890, 1080),
        arrow(1890, 830, 1890, 1080),
    ]

    # Footer formula
    parts += [
        '<rect x="120" y="1210" width="1960" height="90" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>',
        text(1100, 1250, "Training loss:  -log exp(sim(Q,K+)/tau - log q(i+)) / [ exp(pos) + sum_j exp(sim(Q,Nj)/tau - log q(ij)) ]", 24, "700"),
        text(1100, 1282, "Curriculum gradually keeps harder top-k negatives and reduces the global-negative ratio.", 20, "500", color="#475569"),
        "</svg>",
    ]

    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(SVG_PATH)


if __name__ == "__main__":
    main()
