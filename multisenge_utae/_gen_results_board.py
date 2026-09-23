"""Generate RESULTS_BOARD.md from all local metrics JSONs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEG = ROOT.parent / "multisenge_seg"
OUT = ROOT / "results" / "RESULTS_BOARD.md"

C6 = [
    "Dense Built-Up",
    "Sparse Built-Up",
    "Specialized Built-Up",
    "Specialized but Vegetative",
    "Large Scale Networks",
    "Non-urban / other",
]
C10 = [
    "Dense Built-Up",
    "Sparse Built-Up",
    "Specialized Built-Up",
    "Specialized but Vegetative",
    "Large Scale Networks",
    "Arable land",
    "Permanent crops",
    "Grassland",
    "Forest",
    "Water",
]

PAPER6 = {
    "name": "Paper ConvLSTM+Inception",
    "precision": [0.2308, 0.6260, 0.4794, 0.0312, 0.2736, 0.9965],
    "recall": [0.8599, 0.6472, 0.7647, 0.4461, 0.7898, 0.8719],
    "f1": [0.3639, 0.6364, 0.5894, 0.0584, 0.4064, 0.9301],
    "w_p": 0.9591,
    "w_r": 0.8596,
    "w_f1": 0.9018,
    "kappa": 0.4186,
}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def inv_ok(rel: str, filename: str = "test_metrics.json") -> str:
    return "[ok]" if (ROOT / rel / filename).exists() else "[todo]"


def inv_plot(rel: str) -> str:
    return "[ok]" if (ROOT / rel / "history_plot.png").exists() else "[todo]"


def fmt(x, n=4):
    if x is None:
        return "-"
    return f"{float(x):.{n}f}"


def get_list(d: dict | None, key: str, n: int) -> list | None:
    if not d or key not in d or d[key] is None:
        return None
    return list(d[key])[:n]


def section_globals(title: str, rows: list[tuple[str, dict | None]]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Model | Acc | W-P | W-R | W-Sens | W-Spec | **W-F1** | Mean F1 | Mean Sens | Mean Spec | **Kappa** |",
        "|-------|----:|----:|----:|-------:|-------:|---------:|--------:|----------:|----------:|----------:|",
    ]
    for name, d in rows:
        if d is None:
            lines.append(f"| {name} | - | - | - | - | - | [todo] | - | - | - | [todo] |")
            continue
        lines.append(
            "| {name} | {acc} | {wp} | {wr} | {ws} | {wsp} | **{wf}** | {mf} | {ms} | {msp} | **{k}** |".format(
                name=name,
                acc=fmt(d.get("accuracy")),
                wp=fmt(d.get("weighted_precision")),
                wr=fmt(d.get("weighted_recall")),
                ws=fmt(d.get("weighted_sensitivity", d.get("weighted_recall"))),
                wsp=fmt(d.get("weighted_specificity")),
                wf=fmt(d.get("weighted_f1")),
                mf=fmt(d.get("mean_f1")),
                ms=fmt(d.get("mean_sensitivity")),
                msp=fmt(d.get("mean_specificity")),
                k=fmt(d.get("kappa")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def section_per_class(title: str, names: list[str], models: list[tuple[str, dict | None]]) -> str:
    """One big per-class table with P/R/Sens/Spec/F1 columns per model is too wide.
    Emit one subsection per model instead, then a F1 comparison matrix.
    """
    parts = [f"### {title}", ""]
    for name, d in models:
        parts.append(f"#### {name}")
        parts.append("")
        if d is None:
            parts.append("_Missing on laptop - download from PARAM (see section Download)._")
            parts.append("")
            continue
        n = len(names)
        pp = get_list(d, "per_class_precision", n)
        pr = get_list(d, "per_class_recall", n)
        ps = get_list(d, "per_class_sensitivity", n) or pr
        psp = get_list(d, "per_class_specificity", n)
        pf = get_list(d, "per_class_f1", n)
        support = get_list(d, "support", n)
        parts.append(
            "| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |"
        )
        parts.append(
            "|------:|------|----------:|-------:|------------:|------------:|---:|--------:|"
        )
        for i, cname in enumerate(names):
            parts.append(
                "| {i} | {cname} | {p} | {r} | {s} | {sp} | {f} | {sup} |".format(
                    i=i + 1,
                    cname=cname,
                    p=fmt(pp[i] if pp else None),
                    r=fmt(pr[i] if pr else None),
                    s=fmt(ps[i] if ps else None),
                    sp=fmt(psp[i] if psp else None),
                    f=fmt(pf[i] if pf else None),
                    sup=int(support[i]) if support else "-",
                )
            )
        # weighted row
        parts.append(
            "| **W-Avg** | | {p} | {r} | {s} | {sp} | **{f}** | |".format(
                p=fmt(d.get("weighted_precision")),
                r=fmt(d.get("weighted_recall")),
                s=fmt(d.get("weighted_sensitivity", d.get("weighted_recall"))),
                sp=fmt(d.get("weighted_specificity")),
                f=fmt(d.get("weighted_f1")),
            )
        )
        extras = []
        if d.get("accuracy") is not None:
            extras.append(f"Acc {fmt(d['accuracy'])}")
        if d.get("kappa") is not None:
            extras.append(f"Kappa {fmt(d['kappa'])}")
        if d.get("mean_f1") is not None:
            extras.append(f"Mean F1 {fmt(d['mean_f1'])}")
        if extras:
            parts.append("")
            parts.append(" | ".join(extras))
        parts.append("")
    return "\n".join(parts)


def f1_compare(title: str, names: list[str], cols: list[tuple[str, list | None]]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Class | Name | " + " | ".join(c[0] for c in cols) + " |",
        "|------:|------|" + "|".join(["---:"] * len(cols)) + "|",
    ]
    for i, cname in enumerate(names):
        cells = []
        for _, vals in cols:
            cells.append(fmt(vals[i]) if vals else "[todo]")
        lines.append(f"| {i+1} | {cname} | " + " | ".join(cells) + " |")
    # W-F1 / kappa footer as separate note - handled in globals
    lines.append("")
    return "\n".join(lines)


def paper6_as_dict() -> dict:
    return {
        "accuracy": None,
        "weighted_precision": PAPER6["w_p"],
        "weighted_recall": PAPER6["w_r"],
        "weighted_sensitivity": PAPER6["w_r"],
        "weighted_specificity": None,
        "weighted_f1": PAPER6["w_f1"],
        "mean_f1": None,
        "kappa": PAPER6["kappa"],
        "per_class_precision": PAPER6["precision"],
        "per_class_recall": PAPER6["recall"],
        "per_class_sensitivity": PAPER6["recall"],
        "per_class_specificity": None,
        "per_class_f1": PAPER6["f1"],
    }


def main():
    # Load all available
    a4_6 = load(SEG / "results/run_c6_v0/last_test_metrics.json")  # report row
    a4_6_best = load(SEG / "results/run_c6_v0/test_metrics.json")  # best.pt (not report)
    a4_10 = load(SEG / "results/run_c10_v0/test_metrics.json")

    c6_head = load(ROOT / "results/concat_utae/run_c6_head_v0/test_metrics.json")
    c6_full = load(ROOT / "results/concat_utae/run_c6_full_v0/test_metrics.json")
    c10_head = load(ROOT / "results/concat_utae/run_c10_head_v0/test_metrics.json")
    c10_full = load(ROOT / "results/concat_utae/run_c10_full_v0/test_metrics.json")

    ma_head = load(ROOT / "results/ma_utae/ma_c6_gated_head_v0/test_metrics.json")
    ma_full = load(ROOT / "results/ma_utae/ma_c6_gated_full_v0/test_metrics.json")

    s2_head = load(ROOT / "results/concat_utae/run_c6_s2_head_v0/test_metrics.json")
    s2_full = load(ROOT / "results/concat_utae/run_c6_s2_full_v0/test_metrics.json")
    s1_head = load(ROOT / "results/concat_utae/run_c6_s1_head_v0/test_metrics.json")
    s1_full_val = load(ROOT / "results/concat_utae/run_c6_s1_full_v0/best_metrics.json")  # val only
    s1_full = load(ROOT / "results/concat_utae/run_c6_s1_full_v0/test_metrics.json")  # test 31UEQ
    ma_concat_head = load(ROOT / "results/ma_utae/ma_c6_concat_head_v0/test_metrics.json")
    ma_concat_full = load(ROOT / "results/ma_utae/ma_c6_concat_full_v0/test_metrics.json")

    paper6 = paper6_as_dict()

    parts: list[str] = []
    parts.append("# MultiSenGE living results board")
    parts.append("")
    parts.append(
        "Single place for **paper (first-author)**, **A4 ConvLSTM**, **concat U-TAE**, "
        "**MA-UTAE**, and modality ablations. All test numbers = tile **31UEQ** "
        "(geographic split, 4-date S1+S2 unless noted). Metrics from shared "
        "`multisenge_seg/metrics.py` (P / R / Sens / Spec / F1 / W-* / Kappa)."
    )
    parts.append("")
    parts.append(
        "**Legend:** `[ok]` on laptop | `[head]` headline only (JSON still on PARAM) | "
        "`[todo]` not run / missing | `~(val)` = validation only"
    )
    parts.append("")
    parts.append("**Last updated:** 2026-09-16")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 0. Inventory - what is on this laptop")
    parts.append("")
    parts.append("| Run | test_metrics.json | history_plot.png | Notes |")
    parts.append("|-----|:-----------------:|:----------------:|-------|")
    inventory = [
        ("Paper 6c (Table 5)", "-", "-", "Transcribed from RS 2023"),
        ("A4 6c last.pt (report)", "[ok] `multisenge_seg/results/run_c6_v0/last_test_metrics.json`", "-", "Report row"),
        ("A4 6c best.pt", "[ok] `.../test_metrics.json`", "-", "Not the report row"),
        ("A4 10c best.pt", "[ok] `multisenge_seg/results/run_c10_v0/test_metrics.json`", "-", ""),
        ("Concat U-TAE 6c P4", "[ok]", "[ok]", "`results/concat_utae/run_c6_head_v0/`"),
        ("Concat U-TAE 6c P5", "[ok]", "[ok]", "`results/concat_utae/run_c6_full_v0/`"),
        ("Concat U-TAE 10c P4", "[ok]", "[ok]", "`results/concat_utae/run_c10_head_v0/`"),
        ("Concat U-TAE 10c P5", "[ok]", "[ok]", "`results/concat_utae/run_c10_full_v0/`"),
        ("MA gated 6c P4", "[ok]", "[ok]", "`results/ma_utae/ma_c6_gated_head_v0/`"),
        ("MA gated 6c P5", "[ok]", "[ok]", "`results/ma_utae/ma_c6_gated_full_v0/`"),
        ("S2-only 6c P4", "[ok]", "[ok]", "`results/concat_utae/run_c6_s2_head_v0/`"),
        ("S1-only 6c P4", "[ok]", "[ok]", "`results/concat_utae/run_c6_s1_head_v0/`"),
        (
            "S1-only 6c P5",
            inv_ok("results/concat_utae/run_c6_s1_full_v0"),
            inv_plot("results/concat_utae/run_c6_s1_full_v0"),
            "W-F1 0.8970 / κ 0.3537 confirmed (102355 wrote JSON; job exit 9 after write)",
        ),
        (
            "S2-only 6c P5",
            inv_ok("results/concat_utae/run_c6_s2_full_v0"),
            inv_plot("results/concat_utae/run_c6_s2_full_v0"),
            "P5 test **102628**: W-F1 **0.9199** / κ **0.4809** (train 102359 exit 9; early `best.pt`)",
        ),
        (
            "MA concat 6c P4",
            inv_ok("results/ma_utae/ma_c6_concat_head_v0"),
            inv_plot("results/ma_utae/ma_c6_concat_head_v0"),
            "on laptop: W-F1 **0.9218** / κ **0.4961** (P4 > P5)",
        ),
        (
            "MA concat 6c P5",
            inv_ok("results/ma_utae/ma_c6_concat_full_v0"),
            inv_plot("results/ma_utae/ma_c6_concat_full_v0"),
            "on laptop: W-F1 **0.9143** / κ **0.4672** (below gated + concat U-TAE)",
        ),
        ("MA / S1 / S2 10c", "[todo]", "[todo]", "Not started"),
        ("Task H", "[todo]", "[todo]", "Not started"),
        ("P3 probes 6c/10c", "[ok] summaries", "[ok]", "`results/concat_utae/probe_c{6,10}_v0/`"),
    ]
    for row in inventory:
        parts.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    parts.append("")
    parts.append("### 6c closure (2026-09-16)")
    parts.append("")
    parts.append(
        "**All scheduled 6c test JSONs on laptop.** S2 P5: train **102359** FAILED 0:9 (~10h); "
        "eval **102628** COMPLETED using Sep-15 `best.pt` → W-F1 **0.9199**, κ **0.4809** "
        "(slightly above S2 P4 0.9171 / 0.4658). Cancel accidental **102796/102797** if still queued."
    )
    parts.append("")
    parts.append(
        "**W-F1 order:** concat U-TAE P5 (0.9387) > MA gated P5 (0.9353) "
        "> MA concat P4 (0.9218) > S2 P5 (0.9199) > … > S1 P5 (0.8970). **Next:** 10c MA / S1 / S2."
    )
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 1. Headline comparison (paper-facing)")
    parts.append("")
    parts.append("### 1.1 Six-class test (31UEQ)")
    parts.append("")
    parts.append(
        "| Rank | Model | Phase | W-P | W-R | **W-F1** | **Kappa** | Acc | Mean F1 | vs concat P5 W-F1 |"
    )
    parts.append(
        "|-----:|-------|-------|----:|----:|---------:|----------:|----:|--------:|------------------:|"
    )
    def vs_c5(d: dict | None) -> str:
        if d is None or c6_full is None:
            return "[todo]"
        return f"{d['weighted_f1'] - c6_full['weighted_f1']:+.4f}"

    headline_6 = [
        (1, "Concat U-TAE", "P5", c6_full, "-"),
        (2, "MA gated", "P5", ma_full, vs_c5(ma_full)),
        (3, "MA concat fuse", "P4", ma_concat_head, vs_c5(ma_concat_head)),
        (4, "S2-only U-TAE", "P5", s2_full, vs_c5(s2_full)),
        (5, "S2-only U-TAE", "P4", s2_head, vs_c5(s2_head) + " (head)"),
        (6, "MA gated", "P4", ma_head, vs_c5(ma_head)),
        (7, "MA concat fuse", "P5", ma_concat_full, vs_c5(ma_concat_full)),
        (8, "S1-only U-TAE", "P4", s1_head, vs_c5(s1_head)),
        (9, "A4 ConvLSTM (report last.pt)", "-", a4_6, vs_c5(a4_6)),
        (10, "**Paper** ConvLSTM+Inception", "-", paper6, vs_c5(paper6)),
        (11, "Concat U-TAE", "P4", c6_head, vs_c5(c6_head)),
        (12, "S1-only U-TAE", "P5", s1_full, vs_c5(s1_full) + " (worse than S1 P4)"),
    ]
    for rank, model, phase, d, vs in headline_6:
        r = str(rank) if rank else "-"
        if d is None:
            parts.append(f"| {r} | {model} | {phase} | - | - | [todo] | [todo] | - | - | {vs} |")
        else:
            parts.append(
                "| {r} | {model} | {phase} | {wp} | {wr} | **{wf}** | **{k}** | {a} | {mf} | {vs} |".format(
                    r=r,
                    model=model,
                    phase=phase,
                    wp=fmt(d.get("weighted_precision")),
                    wr=fmt(d.get("weighted_recall")),
                    wf=fmt(d.get("weighted_f1")),
                    k=fmt(d.get("kappa")),
                    a=fmt(d.get("accuracy")),
                    mf=fmt(d.get("mean_f1")),
                    vs=vs,
                )
            )
    parts.append("")
    parts.append("### 1.2 Ten-class test (31UEQ)")
    parts.append("")
    parts.append(
        "| Rank | Model | Phase | W-P | W-R | **W-F1** | **Kappa** | Acc | Mean F1 | vs paper W-F1 |"
    )
    parts.append(
        "|-----:|-------|-------|----:|----:|---------:|----------:|----:|--------:|--------------:|"
    )
    # Paper 10c - headlines only from CLASS10.md
    paper10 = {
        "weighted_f1": 0.8851,
        "weighted_recall": 0.8831,
        "kappa": 0.7945,
    }
    headline_10 = [
        (1, "**Paper** ConvLSTM+Inception", "-", paper10, "-"),
        (2, "Concat U-TAE", "P5", c10_full, "-0.0040"),
        (3, "A4 ConvLSTM", "-", a4_10, "-0.0140"),
        (4, "Concat U-TAE", "P4", c10_head, ""),
        (None, "MA gated / concat / S1 / S2", "-", None, "[todo]"),
    ]
    for rank, model, phase, d, vs in headline_10:
        r = str(rank) if rank else "-"
        if d is None:
            parts.append(f"| {r} | {model} | {phase} | - | - | [todo] | [todo] | - | - | {vs} |")
        else:
            parts.append(
                "| {r} | {model} | {phase} | {wp} | {wr} | **{wf}** | **{k}** | {a} | {mf} | {vs} |".format(
                    r=r,
                    model=model,
                    phase=phase,
                    wp=fmt(d.get("weighted_precision")),
                    wr=fmt(d.get("weighted_recall")),
                    wf=fmt(d.get("weighted_f1")),
                    k=fmt(d.get("kappa")),
                    a=fmt(d.get("accuracy")),
                    mf=fmt(d.get("mean_f1")),
                    vs=vs,
                )
            )
    parts.append("")
    parts.append(
        "_Paper 10c per-class P/R/F1 not transcribed in-repo yet (only headlines). "
        "If you paste Table 7 from the PDF, they go here._"
    )
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 2. Six-class - full globals + per-class (everything)")
    parts.append("")
    parts.append(
        section_globals(
            "2.1 Global metrics",
            [
                ("**Paper** ConvLSTM+Inception", paper6),
                ("A4 report `last.pt` ep25", a4_6),
                ("A4 `best.pt` (not report)", a4_6_best),
                ("Concat U-TAE P4 head", c6_head),
                ("Concat U-TAE P5 full", c6_full),
                ("MA gated P4 head", ma_head),
                ("MA gated P5 full", ma_full),
                ("S2-only P4 test", s2_head),
                ("S1-only P4 test", s1_head),
                ("S1-only P5 test", s1_full),
                ("S1-only P5 ~(val) best", s1_full_val),
                ("S2-only P5 test", s2_full),
                ("MA concat P4", ma_concat_head),
                ("MA concat P5", ma_concat_full),
            ],
        )
    )
    parts.append(
        section_per_class(
            "2.2 Per-class Precision / Recall / Sens / Spec / F1",
            C6,
            [
                ("Paper ConvLSTM+Inception (RS 2023 Table 5)", paper6),
                ("A4 ConvLSTM - report `last.pt` (no Spec in JSON)", a4_6),
                ("A4 ConvLSTM - `best.pt` (alt; do not quote as main)", a4_6_best),
                ("Concat U-TAE P4 head", c6_head),
                ("Concat U-TAE P5 full", c6_full),
                ("MA gated P4 head", ma_head),
                ("MA gated P5 full", ma_full),
                ("S2-only P4 test", s2_head),
                ("S2-only P5 test", s2_full),
                ("S1-only P4 test", s1_head),
                ("S1-only P5 test", s1_full),
                ("S1-only P5 ~(val) best (not test)", s1_full_val),
                ("MA concat P4 test", ma_concat_head),
                ("MA concat P5 test", ma_concat_full),
            ],
        )
    )
    parts.append(
        f1_compare(
            "2.3 Per-class F1 side-by-side (paper + ours)",
            C6,
            [
                ("Paper", PAPER6["f1"]),
                ("A4 last", get_list(a4_6, "per_class_f1", 6)),
                ("Concat P4", get_list(c6_head, "per_class_f1", 6)),
                ("Concat P5", get_list(c6_full, "per_class_f1", 6)),
                ("MA P4", get_list(ma_head, "per_class_f1", 6)),
                ("MA P5", get_list(ma_full, "per_class_f1", 6)),
                ("S2 P4", get_list(s2_head, "per_class_f1", 6)),
                ("S1 P4", get_list(s1_head, "per_class_f1", 6)),
                ("S1 P5", get_list(s1_full, "per_class_f1", 6)),
            ],
        )
    )
    parts.append(
        f1_compare(
            "2.4 Per-class Precision side-by-side",
            C6,
            [
                ("Paper", PAPER6["precision"]),
                ("A4 last", get_list(a4_6, "per_class_precision", 6)),
                ("Concat P4", get_list(c6_head, "per_class_precision", 6)),
                ("Concat P5", get_list(c6_full, "per_class_precision", 6)),
                ("MA P4", get_list(ma_head, "per_class_precision", 6)),
                ("MA P5", get_list(ma_full, "per_class_precision", 6)),
            ],
        )
    )
    parts.append(
        f1_compare(
            "2.5 Per-class Recall side-by-side",
            C6,
            [
                ("Paper", PAPER6["recall"]),
                ("A4 last", get_list(a4_6, "per_class_recall", 6)),
                ("Concat P4", get_list(c6_head, "per_class_recall", 6)),
                ("Concat P5", get_list(c6_full, "per_class_recall", 6)),
                ("MA P4", get_list(ma_head, "per_class_recall", 6)),
                ("MA P5", get_list(ma_full, "per_class_recall", 6)),
            ],
        )
    )
    parts.append("---")
    parts.append("")
    parts.append("## 3. Ten-class - full globals + per-class")
    parts.append("")
    parts.append(
        section_globals(
            "3.1 Global metrics",
            [
                ("**Paper** ConvLSTM+Inception (headlines)", paper10),
                ("A4 ConvLSTM", a4_10),
                ("Concat U-TAE P4 head", c10_head),
                ("Concat U-TAE P5 full", c10_full),
                ("MA gated / concat / S1 / S2", None),
            ],
        )
    )
    parts.append(
        section_per_class(
            "3.2 Per-class Precision / Recall / Sens / Spec / F1",
            C10,
            [
                ("Paper ConvLSTM+Inception - Table 7 (need PDF paste)", None),
                ("A4 ConvLSTM (no Spec in JSON)", a4_10),
                ("Concat U-TAE P4 head", c10_head),
                ("Concat U-TAE P5 full", c10_full),
            ],
        )
    )
    parts.append(
        f1_compare(
            "3.3 Per-class F1 side-by-side",
            C10,
            [
                ("A4", get_list(a4_10, "per_class_f1", 10)),
                ("Concat P4", get_list(c10_head, "per_class_f1", 10)),
                ("Concat P5", get_list(c10_full, "per_class_f1", 10)),
            ],
        )
    )
    parts.append(
        f1_compare(
            "3.4 Per-class Precision side-by-side",
            C10,
            [
                ("A4", get_list(a4_10, "per_class_precision", 10)),
                ("Concat P4", get_list(c10_head, "per_class_precision", 10)),
                ("Concat P5", get_list(c10_full, "per_class_precision", 10)),
            ],
        )
    )
    parts.append(
        f1_compare(
            "3.5 Per-class Recall side-by-side",
            C10,
            [
                ("A4", get_list(a4_10, "per_class_recall", 10)),
                ("Concat P4", get_list(c10_head, "per_class_recall", 10)),
                ("Concat P5", get_list(c10_full, "per_class_recall", 10)),
            ],
        )
    )
    parts.append("---")
    parts.append("")
    parts.append("## 4. P3 probes (val only - not paper-facing)")
    parts.append("")
    parts.append("| Setting | Best layer | Val W-F1 | Artifacts |")
    parts.append("|---------|------------|---------:|-----------|")
    parts.append("| 6c linear probes | L2 | **0.7477** | `results/concat_utae/probe_c6_v0/` |")
    parts.append("| 10c linear probes | L1 | **0.8009** | `results/concat_utae/probe_c10_v0/` |")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 5. Honest notes")
    parts.append("")
    parts.append(
        "- **Fair modality match:** S1-only vs paper **ConvLSTM-S1**, S2-only vs **ConvLSTM-S2** "
        "(not vs Inception-S1S2). Full tables: [`PAPER_MODALITY_6CLASS.md`](PAPER_MODALITY_6CLASS.md)."
    )
    parts.append(
        "- **S1 vs paper ConvLSTM-S1 (Table 6 kappa=0.3929, neutral):** "
        "P4 = +0.011 W-F1 / +0.0045 kappa; P5 = -0.003 W-F1 / -0.039 kappa. "
        "Report both; do not soft-word one delta and hype the other."
    )
    parts.append(
        "- **S2 P4 vs paper ConvLSTM-S2 (Table 6 kappa=0.4223):** +0.021 W-F1 / +0.0435 kappa."
    )
    parts.append(
        "- **6c both:** Concat U-TAE **P5** still leads. MA gated P5 is close. "
        "MA concat-fuse peaks at **P4** (0.9218); P5 (0.9143) is lower — full FT did not help that branch."
    )
    parts.append(
        "- **10c:** Concat P5 beats **our** A4 but not the **paper**. Stronger motivation for Task M / Task H."
    )
    parts.append(
        "- Always quote **per-class F1 (esp. 1, 2, 4)** beside W-F1; W-F1 is majority-dominated (class 6 / arable)."
    )
    parts.append(
        "- A4 6c **report** = `last.pt` ep25 (W-F1 0.9037), not `best.pt` (0.9098) - see `TABLE5_TEST_FOR_SIR.md`."
    )
    parts.append(
        "- S1/S2 **paper-facing schedule** = **P5 test**; still show P4 when P5 regresses (S1)."
    )
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 6. Checklist - remaining")
    parts.append("")
    parts.append("- [x] S1 P5 re-test metrics confirmed (102355; ignore exit 9 if JSON written)")
    parts.append("- [x] MA concat 6c **P4+P5 test** on laptop (P4 0.9218 / P5 0.9143)")
    parts.append("- [x] S2-only 6c **P5 test** on laptop (102628; W-F1 0.9199 / κ 0.4809)")
    parts.append("- [ ] **10c** MA gated / concat / S1 / S2 (not started)")
    parts.append("- [ ] Paste **paper Table 7** per-class into section 3")
    parts.append("- [ ] MA gated **10c** P4 -> P5 -> test")
    parts.append("- [ ] Task H on best 6c backbone")
    parts.append("- [ ] Re-run this board generator / update ranks when new JSONs land")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 7. Download from PARAM (metrics + training graphs)")
    parts.append("")
    parts.append("On **Windows laptop** (PowerShell), from `E:\\MTP\\earth2` (or your clone):")
    parts.append("")
    parts.append("```powershell")
    parts.append("# 1) S1/S2 P4 results (full per-class JSON + plots) - HIGHEST PRIORITY for this board")
    parts.append("scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/results/concat_utae/run_c6_s2_head_v0 ./multisenge_utae/results/concat_utae/")
    parts.append("scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/results/concat_utae/run_c6_s1_head_v0 ./multisenge_utae/results/concat_utae/")
    parts.append("")
    parts.append("# 2) S1/S2 training graphs + best_metrics (from checkpoints)")
    parts.append("scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c6_s2_head_v0/history_plot.png ./multisenge_utae/results/concat_utae/run_c6_s2_head_v0/")
    parts.append("scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c6_s1_head_v0/history_plot.png ./multisenge_utae/results/concat_utae/run_c6_s1_head_v0/")
    parts.append("scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c6_s1_full_v0/history_plot.png ./multisenge_utae/results/concat_utae/run_c6_s1_full_v0/")
    parts.append("scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c6_s1_full_v0/best_metrics.json ./multisenge_utae/results/concat_utae/run_c6_s1_full_v0/")
    parts.append("scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c6_s1_full_v0/history.json ./multisenge_utae/results/concat_utae/run_c6_s1_full_v0/")
    parts.append("# if S2 full exists:")
    parts.append("scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c6_s2_full_v0/{history_plot.png,best_metrics.json,history.json} ./multisenge_utae/results/concat_utae/run_c6_s2_full_v0/")
    parts.append("")
    parts.append("# 3) MA concat (when ready)")
    parts.append("scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/ma_c6_concat_head_v0/{history_plot.png,best_metrics.json,history.json} ./multisenge_utae/results/ma_utae/ma_c6_concat_head_v0/")
    parts.append("scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/ma_c6_concat_full_v0/{history_plot.png,best_metrics.json,history.json} ./multisenge_utae/results/ma_utae/ma_c6_concat_full_v0/")
    parts.append("# after P5 test eval:")
    parts.append("scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/results/ma_utae/ma_c6_concat_*_v0 ./multisenge_utae/results/ma_utae/")
    parts.append("")
    parts.append("# 4) Optional: pack everything light (no .pt) on PARAM first")
    parts.append("# On PARAM:")
    parts.append("#   cd ~/MTP/earth2")
    parts.append("#   tar czf /tmp/msge_results_light.tgz \\")
    parts.append("#     multisenge_utae/results \\")
    parts.append("#     multisenge_utae/checkpoints/*/history_plot.png \\")
    parts.append("#     multisenge_utae/checkpoints/*/history.json \\")
    parts.append("#     multisenge_utae/checkpoints/*/best_metrics.json \\")
    parts.append("#     multisenge_utae/checkpoints/*/norm_stats.json")
    parts.append("# Then on laptop:")
    parts.append("scp rihak_iitp@paramrudra.iitp.ac.in:/tmp/msge_results_light.tgz .")
    parts.append("tar xzf msge_results_light.tgz")
    parts.append("```")
    parts.append("")
    parts.append("### Still missing on **this laptop** (already on laptop = skip)")
    parts.append("")
    parts.append("| Path | Why |")
    parts.append("|------|-----|")
    parts.append(
        "| `results/ma_utae/ma_c6_concat_head_v0/test_metrics.json` (+ plot) | P4 test on PARAM only |"
    )
    parts.append(
        "| `results/ma_utae/ma_c6_concat_full_v0/test_metrics.json` (+ plot) | after P5 eval job |"
    )
    parts.append(
        "| `results/concat_utae/run_c6_s2_full_v0/test_metrics.json` (+ plot) | after S2 P5 train+eval |"
    )
    parts.append(
        "| `results/concat_utae/run_c6_s1_full_v0/test_metrics.json` (with CM) | after job **102355** |"
    )
    parts.append(
        "| `results/ma_utae/ma_c6_concat_*` training artifacts | optional: `history.json`, `best_metrics.json` from checkpoints |"
    )
    parts.append("")
    parts.append("**Optional:** paper **Table 7** per-class for 10c paper column")
    parts.append("")
    parts.append("### PARAM check before download")
    parts.append("")
    parts.append("```bash")
    parts.append("cd ~/MTP/earth2")
    parts.append("ls -la multisenge_utae/results/concat_utae/run_c6_s{1,2}_head_v0/")
    parts.append("ls -la multisenge_utae/checkpoints/run_c6_s{1,2}_{head,full}_v0/{history_plot.png,best_metrics.json,best.pt} 2>/dev/null")
    parts.append("ls -la multisenge_utae/checkpoints/ma_c6_concat_{head,full}_v0/{history_plot.png,best.pt} 2>/dev/null")
    parts.append("squeue -u $USER")
    parts.append("```")
    parts.append("")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
