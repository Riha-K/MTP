# MultiSenGE U-TAE results layout

Keep baselines and novelty runs in **separate** subfolders.

| Folder | Contents |
|--------|----------|
| [`RESULTS_BOARD.md`](RESULTS_BOARD.md) | **Living board** — paper + A4 + all P/R/Sens/Spec/F1 + download cmds. Regen: `python multisenge_utae/_gen_results_board.py` |
| [`PAPER_MODALITY_6CLASS.md`](PAPER_MODALITY_6CLASS.md) | Paper ConvLSTM-S1 / S2 / S1S2 / +Inception — **6c (T5/6) + 10c (T7/8)** + fair vs our U-TAE |
| [`../TRAINING_AND_TRANSFER.md`](../TRAINING_AND_TRANSFER.md) | Breast **schedule** vs **PASTIS/ImageNet TL** — why we train U-TAE from scratch on MultiSenGE |
| [`concat_utae/`](concat_utae/) | Frozen **stock concat U-TAE** (P3/P4/P5, 6c + 10c) — baselines for tables |
| [`ma_utae/`](ma_utae/) | **Task M / Task H** MA-UTAE runs |

Do not drop new metrics at the `results/` root (except `RESULTS_BOARD.md`).
