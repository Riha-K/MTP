# multisenge_seg — ConvLSTM+Inception-S1S2 replication

Replicate **Wenger et al. Remote Sensing 2023** on MultiSenGE, then advance under the same protocol.

| Doc | Purpose |
|-----|---------|
| [`PROTOCOL.md`](PROTOCOL.md) | Frozen setting (dates, tiles, classes) |
| [`RESULTS_RS2023_6CLASS.md`](RESULTS_RS2023_6CLASS.md) | 6-class report vs Table 5/6 |
| [`RESULTS_RS2023_10CLASS.md`](RESULTS_RS2023_10CLASS.md) | 10-class report vs Table 7/8 |
| [`TABLE5_TEST_FOR_SIR.md`](TABLE5_TEST_FOR_SIR.md) | Sir-facing 6-class tables |
| [`results/run_c6_v0/last_test_metrics.json`](results/run_c6_v0/last_test_metrics.json) | **6-class report** — last.pt ep 25 |
| [`results/run_c10_v0/test_metrics.json`](results/run_c10_v0/test_metrics.json) | **10-class report** — best.pt |
| `train.py` / `train.sbatch` / `train_c10.sbatch` | PARAM train |
| `eval.sbatch` / `eval_c10.sbatch` | PARAM test eval |

## Data

`LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge/{labels,s1,s2,ground_reference}`

## Train / eval (PARAM)

See [`PARAM_TRANSFER.md`](PARAM_TRANSFER.md).

```bash
sbatch multisenge_seg/train.sbatch          # 6-class → run_c6_v0
sbatch multisenge_seg/train_c10.sbatch      # 10-class → run_c10_v0
sbatch multisenge_seg/eval.sbatch           # 6-class test
sbatch multisenge_seg/eval_c10.sbatch       # 10-class test (best.pt)
```

## Status

6-class and 10-class replicates **frozen**. Optional next: **A5** modern model under the same protocol.
