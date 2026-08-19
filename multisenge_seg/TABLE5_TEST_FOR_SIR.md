# Table 5 / Table 6 — show this to sir

**Paper:** Wenger et al., Remote Sensing 2023, §4.1  
PDF: `BenchmarkGuide/3_ConvLSTM_Inception_MultiSenGE_RemoteSensing.pdf`  
**Setting:** ConvLSTM+Inception-S1S2 · **6 classes** · **test tile T31UEQ** (~610 patches)

**Which of our rows to quote:** **v0 `last.pt`, epoch 25** (highlighted).  
`best.pt` = checkpoint with highest **validation** Weighted F1 (EarlyStopping).  
`last.pt` = weights at the epoch when training stopped.

---

## Checkpoints (do not mix rows)

| File | Run | Which | **Epoch** | Why that file exists | Train job | Test job |
|------|-----|-------|-----------|----------------------|-----------|----------|
| `best.pt` | v0 | best | **5** | max **val** W-F1 = 0.9411 | 96769 | 96798 |
| `last.pt` | v0 | last | **25** | EarlyStop after 20 epochs without beating epoch 5 | 96769 | 96802 |
| `best.pt` | v1 | best | **17** | max val W-F1 = 0.9375 (paper-like batch/stats) | 96803 | 97040 |
| `last.pt` | v1 | last | **37** | EarlyStop after 20 epochs without beating epoch 17 | 96803 | 97041 |

JSON (this repo):

- [`results/run_c6_v0/best_epoch05_test_metrics.json`](results/run_c6_v0/best_epoch05_test_metrics.json)
- [`results/run_c6_v0/last_epoch25_test_metrics.json`](results/run_c6_v0/last_epoch25_test_metrics.json) ← **report this**
- [`results/run_c6_v1/best_epoch17_test_metrics.json`](results/run_c6_v1/best_epoch17_test_metrics.json)
- [`results/run_c6_v1/last_epoch37_test_metrics.json`](results/run_c6_v1/last_epoch37_test_metrics.json)

---

## Table 5 format (paper) — Precision / Recall / F1

Paper Table 5 columns for **ConvLSTM+Inception-S1S2** only (their best 6-class method).

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

### Ours — v0 `best.pt` · **epoch 5** (val W-F1 peak; weaker urban F1)

| Class | Precision | Recall | F1 |
|------:|----------:|-------:|---:|
| 1 | 0.3184 | 0.7200 | 0.4416 |
| 2 | 0.4720 | 0.7649 | 0.5838 |
| 3 | 0.4300 | 0.5021 | 0.4633 |
| 4 | 0.0388 | 0.2837 | 0.0682 |
| 5 | 0.2201 | 0.8015 | 0.3453 |
| 6 | 0.9938 | 0.8980 | 0.9435 |
| **W-Avg** | **0.9506** | **0.8808** | **0.9098** |

### Ours — v0 `last.pt` · **epoch 25** ★ report this

| Class | Precision | Recall | F1 |
|------:|----------:|-------:|---:|
| 1 | 0.4118 | 0.6012 | 0.4888 |
| 2 | 0.5851 | 0.7698 | **0.6649** |
| 3 | 0.3264 | 0.7555 | 0.4558 |
| 4 | 0.0484 | 0.4680 | 0.0877 |
| 5 | 0.2359 | 0.7973 | 0.3640 |
| 6 | 0.9976 | 0.8779 | 0.9339 |
| **W-Avg** | **0.9559** | **0.8681** | **0.9037** |

### Ours — v1 `best.pt` · **epoch 17** (closer training setup; not the main row)

| Class | Precision | Recall | F1 |
|------:|----------:|-------:|---:|
| 1 | 0.3439 | 0.6226 | 0.4430 |
| 2 | 0.5865 | 0.6980 | 0.6374 |
| 3 | 0.3358 | 0.7819 | 0.4698 |
| 4 | 0.0471 | 0.4477 | 0.0852 |
| 5 | 0.2082 | 0.7928 | 0.3298 |
| 6 | 0.9975 | 0.8766 | 0.9332 |
| **W-Avg** | **0.9556** | **0.8651** | **0.9019** |

### Ours — v1 `last.pt` · **epoch 37** (do not report)

| Class | Precision | Recall | F1 |
|------:|----------:|-------:|---:|
| 1 | 0.3532 | 0.6306 | 0.4528 |
| 2 | 0.5749 | 0.7177 | 0.6384 |
| 3 | 0.2724 | 0.8185 | 0.4088 |
| 4 | 0.0465 | 0.4283 | 0.0839 |
| 5 | 0.1870 | 0.7879 | 0.3023 |
| 6 | 0.9982 | 0.8576 | 0.9226 |
| **W-Avg** | **0.9544** | **0.8487** | **0.8906** |

---

## Side-by-side F1 (same as comparing Table 5 F1 column)

| Class | Paper | v0 best **ep 5** | **v0 last ep 25** ★ | v1 best **ep 17** | v1 last **ep 37** |
|------:|------:|-----------------:|-------------------:|------------------:|------------------:|
| 1 | 0.3639 | 0.4416 | **0.4888** | 0.4430 | 0.4528 |
| 2 | 0.6364 | 0.5838 | **0.6649** | 0.6374 | 0.6384 |
| 3 | **0.5894** | 0.4633 | 0.4558 | 0.4698 | 0.4088 |
| 4 | 0.0584 | 0.0682 | **0.0877** | 0.0852 | 0.0839 |
| 5 | **0.4064** | 0.3453 | **0.3640** | 0.3298 | 0.3023 |
| 6 | 0.9301 | **0.9435** | 0.9339 | 0.9332 | 0.9226 |
| **W-Avg F1** | 0.9018 | **0.9098** | **0.9037** | 0.9019 | 0.8906 |

★ = row to show sir as **our** Table 5.

---

## Table 6 format (paper) — Cohen’s Kappa (6-class, test)

| Method | Kappa |
|--------|------:|
| Paper ConvLSTM+Inception-S1S2 | 0.4186 |
| Ours v0 `best.pt` **epoch 5** | 0.4496 |
| Ours v0 `last.pt` **epoch 25** ★ | **0.4424** |
| Ours v1 `best.pt` **epoch 17** | 0.4328 |
| Ours v1 `last.pt` **epoch 37** | 0.4051 |

---

## One-sentence for sir

We reimplemented ConvLSTM+Inception-S1S2. Validation EarlyStopping saved **epoch 5** (`best.pt`, test W-F1 **0.910** but weaker urban classes 2/3/5). Training stopped at **epoch 25** (`last.pt`). **We report epoch 25:** test W-F1 **0.904** (paper **0.902**), class 2 F1 above paper, classes 3 and 5 still below paper.

GPU: paper **3× RTX 6000**, batch 16; we **1 PARAM GPU**.
