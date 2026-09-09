# P3 layer probes — `probe_c6_v0`

**Phase:** P3 (frozen encoder, linear pixel classifier on L0–L3)  
**Eval split:** val tiles **31UFP + 31UGP** (probe fit on train tiles, score on val)  
**Checkpoint:** `multisenge_utae/checkpoints/run_c6_head_v0/best.pt` (P4 head; encoder frozen during P4)

## Expected outputs (after `probe.sbatch`)

| File | Content |
|------|---------|
| `L0_linear_metrics.json` … `L3_linear_metrics.json` | Full per-class metrics per level |
| `probe_summary_linear.json` | W-F1 / kappa per level |
| `probe_summary_linear.md` | Summary table |

Submit on PARAM:

```bash
cd ~/MTP/earth2
git pull
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe_smoke.sbatch   # optional smoke
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe.sbatch
```

Monitor: `tail -f multisenge_utae/artifacts/slurm-probe-<JOBID>.out`

**Plots (sir):** P4/P5 → `plot_history.py` on `history.json` (see README). **Not done yet on PARAM** — run next session. P3 = table in `probe_summary_linear.md` only.
