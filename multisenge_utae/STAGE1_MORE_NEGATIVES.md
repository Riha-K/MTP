# Stage 1 with more negatives

Two extra alignment runs. Same ViT-B/16 student, same InfoNCE temperature 0.07, same projection 256, same AdamW 3e-4, same seed 42, same max of 160 epochs and patience 20. The only change from the finished Stage 1 recipe is the number of negatives, and which frozen S2 encoder is the teacher.

These runs do not overwrite `checkpoints/cmu_s1_vit_v1` or `checkpoints/cmu_s1_vit_gelu_v0`.

## Why

The current Stage 1 batch is 2 patches and 4 dates, so each InfoNCE step has 8 samples. Chance accuracy is 1/8 = 0.125. Chance loss is ln(8) = 2.079. Seven negatives is a small set for the SAR student to learn against.

Each new job keeps those 8 current samples and adds a bank of 56 earlier S2 encoder maps, detached, used only as negatives. The teacher projector is applied to that bank on every step, so the extra keys stay in the same space as the current batch. Storing the projector outputs instead makes those keys stale after the first update, the softmax goes flat, and retrieval locks at 0.125. Every step then has 64 keys. Chance accuracy is 1/64 = 0.0156. Chance loss is ln(64) = 4.159. The training loss will sit higher than the old runs because the chance loss itself is higher. That is the new scale, not a failed alignment.

## How to read the log

`acc=train/val` is retrieval among the current 8 samples. That is the same scale as the ReLU student's 0.416. `student_best.pt` follows the validation half of `acc=`.

`accK=train/val` is retrieval among the 64 keys. It is the harder score. It is not compared with 0.416.

The land-cover comparison, after a later concat head and concat full, is still weighted F1 / kappa on tile 31UEQ.

## The two jobs

| Job script | Teacher | Output |
| --- | --- | --- |
| `train_cmu_relu_neg.sbatch` | `checkpoints/run_c10_s2_full_v0/best.pt` (ReLU, test 0.8865 / 0.7945) | `checkpoints/cmu_s1_vit_relu_neg_v0` |
| `train_cmu_gelu_neg.sbatch` | `checkpoints/run_c10_s2_gelu_full_v0/best.pt` (GELU, test 0.8828 / 0.7800) | `checkpoints/cmu_s1_vit_gelu_neg_v0` |

Submit from the repo directory, one GPU each. This account can hold two GPUs. If job 108542 is still running, only one of these can start.

```bash
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_relu_neg.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_gelu_neg.sbatch
```

Progress:

```bash
grep "acc=" /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-relu-neg-JOBID.out
grep "acc=" /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-gelu-neg-JOBID.out
```

## What comes after

A finished `student_best.pt` is still an alignment file. A 6-class or 10-class land-cover score needs a concat head and then a concat full, then a 31UEQ test. Those jobs are not in these two scripts. The pair to carry forward is the one whose later full test is higher on both 6-class and 10-class. The reference is phase 3 concat full: 0.9379 / 0.5804 and 0.8844 / 0.7861.
