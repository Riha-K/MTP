# Stage 1 with more negatives

Two extra alignment runs. Same ViT-B/16 student, same InfoNCE temperature 0.07, same projection 256, same AdamW 3e-4, same seed 42, same max of 160 epochs and patience 20. The only change from the finished Stage 1 recipe is the number of negatives, and which frozen S2 encoder is the teacher.

These runs do not overwrite `checkpoints/cmu_s1_vit_v1` or `checkpoints/cmu_s1_vit_gelu_v0`.

## Why 8 and 56

The jobs use **8 current samples** and a bank of **56** earlier samples. The key count is 8 + 56 = **64**. A count of 54 was not used. 56 is the smallest number that makes the full key count a power of two and a multiple of 8.

### Why the current set stays at 8

Eight is batch size 2 times the 4 dates already in the Stage 1 recipe (July, August, September, November). It is not a chosen negative count. Every finished Stage 1 run used this batch, including the ReLU student whose validation retrieval is 0.416. One date-sample is a 256 patch through ViT-B/16 and through the frozen U-TAE encoder. That is the memory limit these jobs were written for. Growing the batch so that InfoNCE could see more negatives inside one step would be a second change, on top of the negative bank, and it would move the 0.416 scale.

With 8 samples, InfoNCE has 7 negatives. Chance accuracy is 1/8 = 0.125. Chance loss is ln(8) = 2.079. van den Oord, Li, and Vinyals (2018) define this loss and show that it is a lower bound on mutual information whose ceiling rises with the number of negatives: the bound is at least log(N) minus the loss. Seven negatives keep that ceiling low. That is the reason to add a bank. It is not a reason to change the 8.

### Why the bank has 56, so the dictionary has 64 keys

He, Fan, Wu, Xie, and Girshick (CVPR 2020, MoCo) keep a queue of keys from earlier mini-batches. The queue size is a hyperparameter and is independent of the batch. They do this because a large dictionary cannot be the current batch when GPU memory fixes the batch. Wu, Xiong, Yu, and Lin (CVPR 2018) used a memory bank for the same reason. Chen, Kornblith, Norouzi, and Hinton (ICML 2020, SimCLR) take the other route: they refuse a memory bank and raise the batch from 256 to 8192 so that one step contains up to 16382 negatives. On one GPU, with ViT-B/16 and a U-TAE teacher, the SimCLR route is not available. The queue is the route that fits.

The bank stores detached S2 encoder maps, 56 of them. Fifty-six divides by 8, so seven train batches fill it exactly and the write pointer wraps on a batch boundary. The full dictionary is then 64. Chance accuracy is 1/64 = 0.0156. Chance loss is ln(64) = 4.159. Sixty-four is eight times the old key count, so the task is clearly harder, and the new chance loss cannot be mistaken for the old ln(8).

### Why not a different key count

No paper names 64 as the right dictionary size for SAR to Sentinel-2 alignment. The number is the first step that is clearly larger than 8 and still one change away from the finished recipe. Temperature stays 0.07, the value used by SimCLR and by MoCo v2.

Counts that were set aside:

- **16 or 32 keys.** The dictionary would still be only two or four times the batch. MoCo's point is that the dictionary should be able to grow past the batch. Sixteen keys would leave the task close to the old 7-negative loss.
- **54 bank samples.** Fifty-four does not divide by 8. The last batch of a fill would split across the wrap of the queue. Fifty-six fills in exact batches and lands on 64 keys.
- **100 negatives.** Mitrovic, McWilliams, and Rey (2020, "Less Can Be More in Contrastive Learning") keep the batch fixed and vary only the negative count. On downsampled ImageNet their best linear-probe result was about 100 negatives, and a count as large as the batch of 4096 was worse. One hundred is the same order as our 63 negatives (7 in the batch plus 56 in the bank). We did not copy 100. Their result is the reason a first run stays near that order instead of jumping to tens of thousands.
- **128 or 256 keys.** Both divide cleanly (bank 120 or 248). Memory would still be small. They are the next run if 64 plateaus. They are not the first run, because temperature, learning rate, and the teacher would still be the old ones, and a larger dictionary at temperature 0.07 is a sharper softmax.
- **65536 keys.** That is MoCo's default dictionary, for one vector per ImageNet image and a batch of 256. Our keys are spatial maps, 16 by 16 by 128, because the loss is applied at each ViT site. A bank of 65536 such maps is about 8 GB before the ViT and the U-TAE are loaded. MoCo also needed a momentum encoder to keep a huge queue consistent. We do not add that second mechanism in this run. The same Mitrovic result says a very large negative count can be worse than a moderate one.

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
