# RS 2023 replicate — 6-class test results vs paper

Independent reimplementation of **ConvLSTM+Inception-S1S2** (Wenger et al., *Remote Sensing* 2023, §4.1, Table 5 & 6).

**Quote this row:** v0 **`last.pt`, epoch 25** (not `best.pt` epoch 5).  
Sir-facing tables: [`TABLE5_TEST_FOR_SIR.md`](TABLE5_TEST_FOR_SIR.md) · Word: `results/run_c6_v0/Table5_v0_simple.docx`

| Item | Value |
|------|--------|
| Paper | `BenchmarkGuide/3_ConvLSTM_Inception_MultiSenGE_RemoteSensing.pdf` |
| Split | Test tile **31UEQ** (~610 patches), 6-class |
| Train | PARAM job **96769** (batch 2) |
| Report ckpt | `checkpoints/run_c6_v0/last.pt` · **epoch 25** · eval job **96802** |
| JSON | [`results/run_c6_v0/last_test_metrics.json`](results/run_c6_v0/last_test_metrics.json) |
| Also saved | `best.pt` epoch **5** (val W-F1 peak) · [`test_metrics.json`](results/run_c6_v0/test_metrics.json) |

---

## Table 5 — report row (v0 last.pt, epoch 25)

| Class | Name | Paper P | My P | Paper R | My R | Paper F1 | My F1 |
|-------|------|---------|------|---------|------|----------|-------|
| 1 | Dense Built-Up | 0.2308 | 0.4118 | 0.8599 | 0.6012 | 0.3639 | 0.4888 |
| 2 | Sparse Built-Up | 0.6260 | 0.5851 | 0.6472 | 0.7698 | 0.6364 | 0.6649 |
| 3 | Specialized Built-Up | 0.4794 | 0.3264 | 0.7647 | 0.7555 | 0.5894 | 0.4558 |
| 4 | Specialized but Vegetative | 0.0312 | 0.0484 | 0.4461 | 0.4680 | 0.0584 | 0.0877 |
| 5 | Large Scale Networks | 0.2736 | 0.2359 | 0.7898 | 0.7973 | 0.4064 | 0.3640 |
| 6 | Non-urban / other | 0.9965 | 0.9976 | 0.8719 | 0.8779 | 0.9301 | 0.9339 |
| **W-Avg** | | 0.9591 | 0.9559 | 0.8596 | 0.8681 | **0.9018** | **0.9037** |

Kappa: paper **0.4186** · ours **0.4424**.

Class **2** above paper; class **3 and 5** still below. Class **4** near-zero for both.

---

## Why not quote best.pt (epoch 5)

| | best.pt ep 5 | last.pt ep 25 |
|--|--------------|---------------|
| Test W-F1 | **0.9098** | 0.9037 |
| Class 2 F1 | 0.5838 | **0.6649** |
| Class 5 F1 | 0.3453 | **0.3640** |

`best.pt` is the EarlyStopping pick on **val W-F1** (class 6). `last.pt` is better on urban F1.

---

## v1 (not the main table)

Job **96803**: full-train norm, full class weights, effective batch 16. Test W-F1 **0.9019** (best, ep 17). Class 5 F1 **0.330** — worse than v0 last. JSON: `results/run_c6_v1/`.

---

## Next

6-class **frozen**. Next: **10-class** (paper Table 7/8, `--num-classes 10`) then A5 modern model.
