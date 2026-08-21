# Table 5 / Table 6 — show this to sir

**Paper:** Wenger et al., Remote Sensing 2023, §4.1  
PDF: `BenchmarkGuide/3_ConvLSTM_Inception_MultiSenGE_RemoteSensing.pdf`  
**Setting:** ConvLSTM+Inception-S1S2 · **6 classes** · **test tile T31UEQ** (~610 patches)

**Report row:** v0 **`last.pt`, epoch 25**.  
(`best.pt` epoch 5 = max val W-F1; weaker urban F1 — do not quote as main result.)

JSON: [`results/run_c6_v0/last_test_metrics.json`](results/run_c6_v0/last_test_metrics.json)  
Train job **96769** · test job **96802**

---

## Table 5 — Precision / Recall / F1

### Paper (RS 2023 Table 5)

| Class | Precision | Recall | F1 |
|------:|----------:|-------:|---:|
| 1 Dense Built-Up | 0.2308 | 0.8599 | 0.3639 |
| 2 Sparse Built-Up | 0.6260 | 0.6472 | 0.6364 |
| 3 Specialized Built-Up | 0.4794 | 0.7647 | 0.5894 |
| 4 Specialized but Vegetative | 0.0312 | 0.4461 | 0.0584 |
| 5 Large Scale Networks | 0.2736 | 0.7898 | 0.4064 |
| 6 Non-urban / other | 0.9965 | 0.8719 | 0.9301 |
| **W-Avg** | **0.9591** | **0.8596** | **0.9018** |

### Ours — v0 `last.pt` · **epoch 25**

| Class | Precision | Recall | F1 |
|------:|----------:|-------:|---:|
| 1 | 0.4118 | 0.6012 | 0.4888 |
| 2 | 0.5851 | 0.7698 | **0.6649** |
| 3 | 0.3264 | 0.7555 | 0.4558 |
| 4 | 0.0484 | 0.4680 | 0.0877 |
| 5 | 0.2359 | 0.7973 | 0.3640 |
| 6 | 0.9976 | 0.8779 | 0.9339 |
| **W-Avg** | **0.9559** | **0.8681** | **0.9037** |

---

## Table 6 — Cohen’s Kappa (6-class, test)

| Method | Kappa |
|--------|------:|
| Paper ConvLSTM+Inception-S1S2 | 0.4186 |
| Ours v0 `last.pt` **epoch 25** | **0.4424** |

---

## One-sentence for sir

Independent reimplementation of ConvLSTM+Inception-S1S2 on MultiSenGE (test 31UEQ). Report **epoch 25** (`last.pt`): W-F1 **0.904** (paper **0.902**), kappa **0.442** (paper **0.419**). Class 2 above paper; classes 3 and 5 still below. GPU: paper 3× RTX 6000 batch 16; we 1 PARAM GPU.
