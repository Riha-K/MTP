# Lab GPU transfer — `172.30.1.70` (riha_2511ai47)

**Host:** `172.30.1.70` · **User:** `riha_2511ai47`  
**Not PARAM** — usually no `sbatch`; train with `tmux` + `python` (or whatever they use).

**Do not** git-push rasters. Copy data with `scp`/`rsync` only.

Laptop source: `e:\MTP\earth2\`  
Remote home target (suggested): `~/MTP/earth2/`

---

## Step 0 — On the GPU box (you are already logged in)

```bash
hostname
whoami
nvidia-smi                  # confirm GPU
df -h $HOME /              # need ~200+ GB free if copying S1+S2
which python3; python3 --version
which git
mkdir -p ~/MTP
cd ~/MTP
```

Paste `df -h` and `nvidia-smi` output before starting the big copy.

---

## Step 1 — Code via Git (preferred; small)

On the **GPU box**:

```bash
cd ~/MTP
# if empty:
git clone https://github.com/Riha-K/MTP.git earth2
cd earth2
git pull origin main
ls multisenge_seg LULCDial-s1
```

If `git` is blocked from GitHub, from **laptop PowerShell**:

```powershell
scp -r e:\MTP\earth2\multisenge_seg riha_2511ai47@172.30.1.70:~/MTP/earth2/
# (create ~/MTP/earth2 first on remote)
```

Better: clone once; then only sync data folders.

---

## Step 2 — Create data dirs on GPU box

```bash
MS=~/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge
mkdir -p "$MS"
```

---

## Step 3 — Laptop → GPU: small data first

**PowerShell on laptop** (keep window open):

```powershell
$H = "riha_2511ai47@172.30.1.70"
$L = "e:\MTP\earth2\LULCDial-s1\data\lulcdial_s1\ai4lcc\multisenge"
$R = "/home/riha_2511ai47/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"

ssh $H "mkdir -p $R"

# labels (~minutes)
scp -r "$L\labels" "${H}:${R}/"

# ground_reference
scp -r "$L\ground_reference" "${H}:${R}/"
```

Verify on GPU box:

```bash
MS=~/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge
find "$MS/labels" -type f | wc -l            # 8157
find "$MS/ground_reference" -type f | wc -l  # 8157
```

---

## Step 4 — Large rasters (overnight)

| Folder | ~Size | Need? |
|--------|-------|--------|
| `s2/` | ~88 GB | **Yes** (CNN) |
| `s1/` | ~110 GB | **Yes** (CNN multimodal) |

```powershell
$H = "riha_2511ai47@172.30.1.70"
$L = "e:\MTP\earth2\LULCDial-s1\data\lulcdial_s1\ai4lcc\multisenge"
$R = "/home/riha_2511ai47/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"

# Start S2 first (required). Leave PC on / sleep disabled.
scp -r "$L\s2" "${H}:${R}/"

# Then S1
scp -r "$L\s1" "${H}:${R}/"
```

**Resumable (if you have WSL):**

```bash
H=riha_2511ai47@172.30.1.70
L=/mnt/e/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge
R=/home/riha_2511ai47/MTP/earth2/LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge

rsync -avh --progress --partial "$L/s2/" "$H:$R/s2/"
rsync -avh --progress --partial "$L/s1/" "$H:$R/s1/"
```

---

## What **not** to copy

- `LULCDial-s1/checkpoints/` (old VLM weights — optional later)
- `LULCDial-s1/data/.../shards/` (VLM shards — not needed for CNN)
- `.venv`, `__pycache__`, `writeup/` unless you want them
- Whole `EarthDial` weights unless doing VLM again

For **CNN validation** you only need: **repo + labels + GR + s1 + s2**.

---

## Step 5 — Env on GPU box (after code exists)

```bash
cd ~/MTP/earth2
python3 -m venv ~/venvs/msge
source ~/venvs/msge/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # adjust CUDA as needed
pip install numpy rasterio
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Ask the lab admin which CUDA/`module` they use if `pip` torch fails.

---

## Step 6 — Smoke + train (no Slurm)

```bash
cd ~/MTP/earth2
source ~/venvs/msge/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m multisenge_seg.smoke_index --max-labels 400

# keep job alive after SSH drop:
tmux new -s msge
python -m multisenge_seg.train \
  --max-train 4 --max-val 2 --epochs 1 --batch-size 1 --workers 0 \
  --out-dir multisenge_seg/checkpoints/smoke_c6
# Ctrl-b then d  to detach

# Full run later:
python -m multisenge_seg.train \
  --num-classes 6 --epochs 80 --batch-size 2 \
  --out-dir multisenge_seg/checkpoints/run_c6_v0
```

---

## Checklist

- [ ] `nvidia-smi` OK  
- [ ] Disk free enough (~200 GB for S1+S2)  
- [ ] `git clone` / `pull` → `multisenge_seg/` present  
- [ ] labels 8157 · GR 8157  
- [ ] s2 copying / done  
- [ ] s1 copying / done  
- [ ] venv + torch + rasterio  
- [ ] smoke train in `tmux`
