# CMU phase: what we built, and why it changed

Use this note when explaining Stage 1 (and the coded Stage 2) to sir.
Short version first, then the detail.

## Say this first

We are aligning Sentinel-1 to a frozen Sentinel-2 U-TAE, the way MM-OVSeg aligns SAR to a frozen image teacher, but our teacher is our own MultiSenGE S2 model and there is no CLIP and no text.

One date of S2 (10 bands) goes through the frozen S2 U-TAE encoder. The same patch and the same date of S1 (VV, VH) goes through a ViT-B/16. InfoNCE pulls those two features together. Time is not mixed here. The four dates stay separate so that July SAR is matched to July S2.

Stage 2 stacks the four dates, runs L-TAE on both streams, fuses them, and trains a closed-set land-cover head (6 classes and 10 classes). The S2 side of that head is the already-trained 10-class S2 U-TAE, not a new random U-TAE. See the section below.

## What goes into the loss

| | S2 teacher | S1 student |
|---|---|---|
| Input | 10 bands, 256x256 | VV+VH, 256x256 |
| Network | Frozen U-TAE encoder from `run_c10_s2_full_v0` | ViT-B/16, ImageNet init, 2-channel stem |
| Time | No L-TAE in Stage 1 | No L-TAE in Stage 1 |
| Dates | Jul, Aug, Sep, Nov (4) | Same four dates, same patch |
| Output now | 16x16 map, 128 channels, resized from the 32x32 bottleneck | 16x16 patch tokens, 768-d |
| Projector | 128 to 256, Linear-LayerNorm-GELU-Linear | 768 to 256, same |
| Loss | InfoNCE, temperature 0.07 | Same |

A training step uses batch 2, so InfoNCE sees 2 x 4 = 8 samples. Chance loss is ln(8) = 2.079. Chance accuracy is 1/8 = 0.125. If the printed train loss stays at 2.079, the student did not learn.

The positive pair is always the same patch and the same date. The other samples in the step are the negatives.

## Where the code lives

| Piece | File |
|---|---|
| ViT-B/16, 2-channel stem, 256 image | `multisenge_utae/models/s1_vit.py` |
| Projectors and InfoNCE | `multisenge_utae/models/cmu.py` |
| Stage 1 training loop | `multisenge_utae/train_cmu.py` |
| Smoke job (8 train / 4 val patches, 1 epoch) | `multisenge_utae/train_cmu_smoke.sbatch` |
| Full job (80 epochs, patience 20) | `multisenge_utae/train_cmu.sbatch` |
| Stage 2 model (not trained yet) | `multisenge_utae/models/cmu_vit_utae.py` |
| Stage 2 training | `multisenge_utae/train_cmu_vit.py` |
| Frozen S2 weights | `multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt` |

Normalization is the same mean and std the S2 U-TAE was trained with. We do not re-estimate it.

## What failed, and what we changed

### 1. Teacher projector was not learning

Job **105732** (cancelled after 6 epochs). Loss stayed at 2.0794 and accuracy at 0.125.

The S2 encoder is frozen, which is correct. The teacher projector was also blocked, because its input was detached. The student then had no useful target to follow.

Change: the encoder stays in `no_grad`. The teacher projector is outside that block, so it receives gradients. The ViT also starts from ImageNet weights (the 2-channel stem is the mean of the RGB filters), not from a random network. Position embeddings are stretched from the ImageNet 14x14 grid to 16x16 so a 256 patch fits ViT-B/16.

### 2. The ViT still believed the image was 224

Job **105801** failed in the first step: `Wrong image height! Expected 224 but got 256`.

We had resized the position table to 256, but torchvision checks its own `image_size`, which was still 224.

Change: after the resize, set `vit.image_size` to 256.

### 3. One averaged S2 vector gave InfoNCE nothing to match

Job **105986** (28 Sep 2026, 07:13, ragpu006) actually finished an epoch. Train loss was 2.0793. That is still chance. The smoke is written to exit 1 in that case, so the 80-epoch job does not start. Job **105987** was cancelled.

The S2 encoder outputs a 32x32 map. We were averaging that map into one vector per date. After that average, different fields look almost the same, so the positive and the negatives are the same point. InfoNCE cannot pull the SAR token toward a distinct optical target.

Change, used in smoke **105990** and full job **105991**: keep the map. Resize the S2 bottleneck from 32x32 to 16x16. Use every ViT patch token, not only the CLS token. At each of the 256 sites, InfoNCE still compares the 8 samples. The positive is still the same patch and the same date, but only at that site. Other sites are not used as negatives, so chance loss is still 2.079.

### 4. The projector died, then the loss could not move

Job **105991** (started about 03:30 on 29 Sep 2026, `racn116`) did learn at first. Epoch 5 val loss was **2.005** and val accuracy **0.189**. From epoch 8 through 16 the log is identical: train/val loss **2.0794 / 2.0787**, accuracy **0.125 / 0.125**. That is chance on every batch, so the weights had stopped changing the match.

The projector was Linear, ReLU, Linear. Once the ReLU hidden layer is zero for every sample, every date becomes the same vector. InfoNCE is a tie, the loss is `ln(8)`, and the gradient is zero. The run cannot leave that state. Cancel it. Do not use `cmu_s1_vit_v0` as the Stage 2 student.

Change: the projector is Linear, LayerNorm, GELU, Linear, so a dead ReLU cannot wipe the teacher target. The next full job writes `checkpoints/cmu_s1_vit_v1`.

## How to read the next smoke

Log: `multisenge_utae/logs/slurm-cmu-smoke-105990.out`.

- The log must contain `spatial InfoNCE`. If it still says `InfoNCE N≈batch*4`, that is the old pooled run.
- Pass: train loss below 2.079. Then full job **105991** is allowed to start (`afterok`).
- Fail: loss still about 2.079. The full job must not run. Do not use `cmu_s1_vit_smoke` from 105986 or the cancelled 105732 checkpoint.

## Why Stage 2 loads the trained S2 encoder

Earlier U-TAE and MA-UTAE runs start **basic**: random weights, no older checkpoint. P4 freezes the encoder and trains the head. P5 loads that same P4 file and trains the whole network. A 6-class run never loads a 10-class checkpoint, and a 10-class run never loads a 6-class checkpoint. The 10-class S2 model was chosen as the CMU teacher only. That choice does not mean the old land-cover runs mixed class counts.

CMU Stage 2 is the exception. Job **106824** (6-class head) and the 10-class head both copy the encoder and the L-TAE from `run_c10_s2_full_v0`. They do not copy that file's 10-class classifier. The 6-class (or 10-class) decoder is new. The old CONCAT 6-class P4 weights (`run_c6_head_v0`) are a score to compare against. They are not loaded.

The trained encoder stays because Stage 1 matched the ViT to that encoder's maps. A basic S2 encoder would be a different set of maps, and the ViT would be aligned to a teacher that is no longer in the network.

| Run | What the network is | Where the weights start |
|---|---|---|
| Paper ConvLSTM, and ConvLSTM+Inception | MultiSenGE paper models. Inception is their extra block on the ConvLSTM. | Published numbers. Those weights are not loaded into our models. |
| A4 ConvLSTM | Our ConvLSTM, same idea as the paper. | Basic. Its own training. |
| U-TAE, S1+S2, 6c and 10c | One U-TAE. SAR and optical bands are stacked into one input. | Basic at P4. P5 continues that P4 file. |
| U-TAE, S2 only, 6c and 10c | Same U-TAE, optical bands only. | Basic at P4. P5 continues that P4. The 10c P5 of this line is `run_c10_s2_full_v0`. |
| U-TAE, S1 only, 6c and 10c | Same U-TAE, SAR bands only. | Basic at P4. P5 continues that P4. |
| MA-UTAE, gated and CONCAT, 6c and 10c | Two U-TAE encoders, one SAR and one optical, then a fusion. | Basic at P4. P5 continues that P4. 6c never loads a 10c file. |
| CMU Stage 2, 6c and 10c | SAR is the Stage 1 ViT. Optical is the trained 10c S2 encoder and L-TAE. New land-cover decoder. | `cmu_s1_vit_v1/student_best.pt` and `run_c10_s2_full_v0`. Not basic. |

## Stage 2, now training

After a real `student_best.pt` exists:

1. S1: the CMU ViT, four dates, then a new S1 L-TAE.
2. S2: the same U-TAE encoder and L-TAE as `run_c10_s2_full_v0`, not a new random encoder.
3. Fuse at the bottleneck only (CONCAT). The decoder uses S2 skip connections, not S1 skips, because the ViT is one scale and the U-TAE is a pyramid.
4. P4 freezes the loaded S2 encoder, the S2 L-TAE, and the CMU ViT. It trains the S1 L-TAE (Stage 1 has no L-TAE), the adapter, the fusion, and the decoder.
5. P5 fine-tunes everything from the P4 checkpoint.
6. Order: 6-class P4 then P5, then 10-class. Test tile 31UEQ is a separate job (`eval_cmu_vit.sbatch`). The training log's weighted F1 is validation, not that test.

Job **106245** finished all 80 epochs on 1 Oct 2026. Final train accuracy **0.601**, validation accuracy **0.412**. The saved best is epoch 70, validation accuracy **0.416**, in `checkpoints/cmu_s1_vit_v1/student_best.pt`. Use that file for Stage 2. `student_last.pt` is epoch 80, where validation accuracy had already stopped rising.

Do not start Stage 2 from `cmu_s1_vit_v0` or from the cancelled 105732 checkpoint.
