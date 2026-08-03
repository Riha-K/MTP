# How to publish SAR-LC-Bench (separate public repo)

Your research repo (`MTP` / earth2) contains personal notes (`log.md`, `ROADMAP.md`, PARAM paths, drafts).
**Do not make that whole repo the public bench.**

## Current plan (2026-08-03)

**Defer** creating the public GitHub repo and Zenodo TIFF upload until **after Phase 3 (bi-temporal lite)**.
Reason: one public release can include single-date GE test pack + ~100-patch 2-date pack, so users do not download twice.

Until then, keep drafting in this research tree under `sar_lc_bench_v0.1/`.

## Recommendation (when you do publish)

| Repo | Visibility | Contents |
|------|------------|----------|
| **This research repo** (`MTP`) | Private or as-is | Training code, PARAM notes, writeup, experiments |
| **New public repo** e.g. `SAR-LC-Bench` | **Public** | Only `sar_lc_bench_v0.1/` contents (+ optional slim eval copy) |
| **Zenodo** (free) | Public | Compact TIFF zips (~2497 GE test + bi-temporal pack) — **not** full 110 GB AI4LCC |

## Steps when you are ready to go public

1. On GitHub: **New repository** → name `SAR-LC-Bench` (or similar) → Public → **no** README (empty).
2. On your laptop:

```bash
# copy only the clean package
mkdir -p ~/SAR-LC-Bench
cp -r /path/to/earth2/sar_lc_bench_v0.1/* ~/SAR-LC-Bench/

cd ~/SAR-LC-Bench
git init
git add .
git commit -m "Initial SAR-LC-Bench v0.1 protocol and leaderboard"
git branch -M main
git remote add origin https://github.com/<you>/SAR-LC-Bench.git
git push -u origin main
```

3. Upload **large** TIFF packs via **Zenodo** (preferred DOI) or GitHub Release — not via `git add` of tens of GB.
4. Optionally mirror JSONL + leaderboard on **Hugging Face Datasets**.
5. In the paper: link the public GitHub URL + Zenodo DOI.

## What not to publish from MTP

- `log.md`, student meeting notes, passwords  
- PARAM usernames / absolute `/home/...` secrets beyond public sbatch examples  
- Full `writeup/` until you choose to release them  
- Full MultiSenGE `s1/` (~110 GB+) inside git  

## Timing vs paper

Protocol + leaderboard + eval script can go public with or without journal acceptance.
Current team choice: **wait until Phase 3 data exists**, then one clean release.
