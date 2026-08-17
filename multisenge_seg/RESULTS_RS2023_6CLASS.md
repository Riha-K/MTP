# RS 2023 replicate — 6-class test results vs paper

Independent reimplementation of **ConvLSTM+Inception-S1S2** (Wenger et al., *Remote Sensing* 2023, §4.1, Table 5 & 6).

| Item | Value |
|------|--------|
| Paper | `BenchmarkGuide/3_ConvLSTM_Inception_MultiSenGE_RemoteSensing.pdf` |
| Protocol | `PROTOCOL.md` |
| Metrics code | `metrics.py` → `scores_from_cm()` |
| Train job | PARAM **96769** → `checkpoints/run_c6_v0/best.pt` (epoch 5, val wF1 0.9411) |
| Test eval | PARAM **96798** |
| **JSON (in this repo)** | [`results/run_c6_v0/test_metrics.json`](results/run_c6_v0/test_metrics.json) |
| JSON (PARAM only, large ckpt dir) | `~/MTP/earth2/multisenge_seg/checkpoints/run_c6_v0/test_metrics.json` |
| Split | **Test** tile **31UEQ** (~610 patches), 6-class merge |
| Paper row | ConvLSTM+Inception-S1S2 (last column in Table 5) |

Legend for **Δ vs paper**: **above** = my score higher · **below** = my score lower · **≈** = within 0.01

---

## Table 5 — Precision / Recall / F1 (test zone)

| Class | Name | Paper P | My P | Δ P | Paper R | My R | Δ R | Paper F1 | My F1 | Δ F1 |
|-------|------|---------|------|-------|---------|------|-------|----------|-------|------|
| 1 | Dense Built-Up | 0.2308 | 0.3184 | **above** +0.088 | 0.8599 | 0.7200 | below −0.140 | 0.3639 | 0.4416 | **above** +0.078 |
| 2 | Sparse Built-Up | 0.6260 | 0.4720 | below −0.154 | 0.6472 | 0.7649 | **above** +0.118 | 0.6364 | 0.5838 | below −0.053 |
| 3 | Specialized Built-Up | 0.4794 | 0.4300 | below −0.049 | 0.7647 | 0.5021 | below −0.263 | 0.5894 | 0.4633 | below −0.126 |
| 4 | Specialized but Vegetative | 0.0312 | 0.0388 | **above** +0.008 | 0.4461 | 0.2837 | below −0.162 | 0.0584 | 0.0682 | **above** +0.010 |
| 5 | Large Scale Networks | 0.2736 | 0.2201 | below −0.054 | 0.7898 | 0.8015 | **above** +0.012 | 0.4064 | 0.3453 | below −0.061 |
| 6 | Non-urban / other | 0.9965 | 0.9938 | ≈ −0.003 | 0.8719 | 0.8980 | **above** +0.026 | 0.9301 | 0.9435 | **above** +0.013 |
| **W-Avg** | | **0.9591** | **0.9506** | ≈ −0.009 | **0.8596** | **0.8808** | **above** +0.021 | **0.9018** | **0.9098** | **above** +0.008 |

**Pattern (same as paper):** urban classes show **Recall > Precision** (model finds pixels but confuses labels). Class **4** remains near-zero F1 for both.

---

## Table 6 — Cohen's Kappa (6-class, test)

| | Paper | Mine | Δ |
|--|-------|------|---|
| Kappa | 0.4186 | **0.4496** | **above** +0.031 |

---

## Headline summary

| Metric | Paper | Mine | Verdict |
|--------|-------|------|---------|
| Weighted F1 | 0.9018 | **0.9098** | **above** (+0.8 pp) |
| Weighted Recall | 0.8596 | **0.8808** | **above** |
| Weighted Precision | 0.9591 | 0.9506 | below (small) |
| Kappa | 0.4186 | **0.4496** | **above** |
| Accuracy | ~0.86 (paper W-R) | **0.8808** | **above** |

**Conclusion:** 6-class replicate **matches or beats** the published best model on the paper test tile (Weighted F1 primary metric).

---

## Validation (not in paper Table 5)

| Split | Weighted F1 | Kappa | Note |
|-------|-------------|-------|------|
| Val (31UFP+31UGP) | 0.9411 | 0.448 | epoch-5 `best.pt`; paper does not report val |

---

## Where `test_metrics.json` lives

Eval writes a **small JSON** (scores only, not the model). Two copies:

| Copy | Path | In git? |
|------|------|---------|
| **This repo** | [`multisenge_seg/results/run_c6_v0/test_metrics.json`](results/run_c6_v0/test_metrics.json) | yes |
| PARAM (after eval) | `~/MTP/earth2/multisenge_seg/checkpoints/run_c6_v0/test_metrics.json` | no (`best.pt` is large; keep on PARAM) |

```bash
# PARAM original
cat ~/MTP/earth2/multisenge_seg/checkpoints/run_c6_v0/test_metrics.json

# laptop / git
cat multisenge_seg/results/run_c6_v0/test_metrics.json
```

Re-run eval: `sbatch multisenge_seg/eval.sbatch` — output table comes from `format_prf_table()` in `metrics.py`.
