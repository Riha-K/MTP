# PARAM transfer — MultiSenGE CNN validation

**Goal:** Get **code + MultiSenGE rasters** onto PARAM so you can `sbatch` train.  
**Do not** put `s1/` / `s2/` / `ground_reference/` in git.

**PARAM login (from RUNBOOK):** `ssh rihak_iitp@paramrudra.iitp.ac.in`  
**Code root on PARAM:** `/home/rihak_iitp/MTP/earth2`  
**Data root (target):**  
`/home/rihak_iitp/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge/`

Laptop data (source):  
`e:\MTP\earth2\LULCDial-s1\data\lulcdial_s1\ai4lcc\multisenge\`

---

## Sizes (approx)

| Folder | Size | Notes |
|--------|------|--------|
| code (`git pull`) | small | Already on GitHub `main` |
| `labels/` | ~few MB | 8157 JSON — transfer first |
| `ground_reference/` | ~small–few GB | 8157 TIFFs — transfer second |
| `s1/` | **~110 GB** | May already exist on PARAM from LULCDial |
| `s2/` | **~88 GB** | Likely **new** — longest copy |

Overnight / background transfer is normal for S1+S2.

---

## Step 0 — On PARAM: disk + what already exists

```bash
ssh rihak_iitp@paramrudra.iitp.ac.in
# CAPTCHA + password

df -h $HOME
mkdir -p ~/MTP/earth2
cd ~/MTP/earth2

# Code
git status || git clone https://github.com/Riha-K/MTP.git .
git pull origin main

MS=~/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge
mkdir -p "$MS"
ls -la "$MS"
# Count if present (fast):
for d in labels s1 s2 ground_reference; do
  if [ -d "$MS/$d" ]; then echo -n "$d: "; find "$MS/$d" -type f | wc -l; else echo "$d: MISSING"; fi
done
du -sh "$MS"/labels "$MS"/s1 "$MS"/s2 "$MS"/ground_reference 2>/dev/null
```

**Expected for train:**

| Dir | Target count (approx) |
|-----|------------------------|
| `labels` | 8157 |
| `ground_reference` | 8157 |
| `s1` | ~209k tifs (or whatever your laptop has) |
| `s2` | ~72k tifs |

If **`s1` already full** from LULCDial, **do not re-copy** it.

---

## Step 1 — Laptop: pull is enough for code

Code is already pushed. On PARAM only `git pull` (Step 0). No need to `scp` the repo if clone exists.

---

## Step 2 — Laptop → PARAM: small folders first

Open **PowerShell** on the laptop (new window; leave it open while copying).

```powershell
$PARAM = "rihak_iitp@paramrudra.iitp.ac.in"
$LOCAL = "e:\MTP\earth2\LULCDial-s1\data\lulcdial_s1\ai4lcc\multisenge"
$REMOTE = "/home/rihak_iitp/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"

# Ensure remote parent exists
ssh $PARAM "mkdir -p $REMOTE"

# 1) labels (~minutes)
scp -r "$LOCAL\labels" "${PARAM}:${REMOTE}/"

# 2) ground_reference (~tens of minutes)
scp -r "$LOCAL\ground_reference" "${PARAM}:${REMOTE}/"
```

Verify on PARAM:

```bash
MS=~/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge
find "$MS/labels" -type f | wc -l          # 8157
find "$MS/ground_reference" -type f | wc -l  # 8157
```

---

## Step 3 — Large rasters (`s2`, and `s1` if missing)

**Prefer `rsync` if available** (resumable). On Windows, OpenSSH `scp` works; interrupt and re-run may duplicate — prefer one long overnight copy.

### Option A — `scp` (simple)

```powershell
$PARAM = "rihak_iitp@paramrudra.iitp.ac.in"
$LOCAL = "e:\MTP\earth2\LULCDial-s1\data\lulcdial_s1\ai4lcc\multisenge"
$REMOTE = "/home/rihak_iitp/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"

# Only if PARAM s1 count is far below laptop:
# scp -r "$LOCAL\s1" "${PARAM}:${REMOTE}/"

# S2 (required for CNN) — overnight
scp -r "$LOCAL\s2" "${PARAM}:${REMOTE}/"
```

### Option B — WSL / Git Bash `rsync` (better resume)

```bash
PARAM=rihak_iitp@paramrudra.iitp.ac.in
LOCAL=/mnt/e/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge
REMOTE=/home/rihak_iitp/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge

rsync -avh --progress --partial "$LOCAL/s2/" "$PARAM:$REMOTE/s2/"
# if needed:
# rsync -avh --progress --partial "$LOCAL/s1/" "$PARAM:$REMOTE/s1/"
```

---

## Step 4 — PARAM env deps (once)

CNN train needs **torch** (module) + **rasterio** / **numpy**.

```bash
module purge
module load MLDL/Pytorch-gpu
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install --user rasterio   # if missing
python -c "import rasterio; print(rasterio.__version__)"
```

---

## Step 5 — Smoke (after data lands)

```bash
cd ~/MTP/earth2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
module purge
module load MLDL/Pytorch-gpu

# Quick index (partial)
python -m multisenge_seg.smoke_index --max-labels 400

# Tiny train smoke (needs a few real patches + GPU node via salloc or short sbatch)
python -m multisenge_seg.train \
  --max-train 4 --max-val 2 --epochs 1 --batch-size 1 --workers 0 \
  --out-dir multisenge_seg/checkpoints/smoke_c6
```

Full job:

```bash
cd ~/MTP/earth2
sbatch multisenge_seg/train.sbatch
squeue -u $USER
```

---

## Checklist

- [ ] PARAM: `git pull` → `multisenge_seg/` present  
- [ ] Disk free ≳ **200 GB** if copying both S1+S2 (or ≳ **100 GB** if S1 already there + only S2)  
- [ ] `labels` 8157  
- [ ] `ground_reference` 8157  
- [ ] `s1` present (reuse or copy)  
- [ ] `s2` present (~72k)  
- [ ] `module load MLDL/Pytorch-gpu` + `rasterio`  
- [ ] smoke_index + tiny train  
- [ ] `sbatch multisenge_seg/train.sbatch`

---

## Notes

- Login nodes: use for `scp`/`rsync`/`git`/`pip`. Heavy train → **`sbatch`**, not interactive overnight.  
- CAPTCHA: each new SSH/`scp` may ask again; keep one long `scp` session.  
- If home quota is tight, ask PARAM admins / sir for a scratch project path and symlink `multisenge` there.
