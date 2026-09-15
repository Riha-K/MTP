# U-TAE training schedule vs transfer learning (future reference)

**Last updated:** 2026-09-14

## Question

The breast-cancer paper (Singh et al., TCBB 2021) is a **transfer-learning** study. We follow their **P3 / P4 / P5** workflow on MultiSenGE. Why do we **not** load PASTIS, ImageNet, or other external pretrained weights into U-TAE?

## What the breast paper actually transfers

- A **large encoder already trained** on a related source domain (e.g. ImageNet / VGG-style natural images).
- Then: layer probes (which layer is useful?) → train head only → full fine-tune.
- "Transfer learning" there = **reuse those pretrained weights** + a careful fine-tune recipe.

## What we borrowed vs what we did not

| Borrowed from breast-style workflow | Did **not** borrow |
|-------------------------------------|--------------------|
| P3: frozen encoder + linear/RF probes on val | PASTIS / ImageNet / any external checkpoint |
| P4: freeze encoder + L-TAE; train decoder/head | Cross-dataset weight transfer |
| P5: unfreeze all; `--init-ckpt` from **our** P4 `best.pt` | Claiming "breast TL for remote sensing" as the main novelty |

P5 `--init-ckpt` is **continued training on the same dataset** (MultiSenGE), not transfer from another corpus.

## Why no PASTIS → MultiSenGE transfer

1. **Fair comparison with MultiSenGE paper and A4**  
   ConvLSTM+Inception (Wenger RS 2023) and our **A4** replicate are trained **from scratch** on MultiSenGE. Initializing U-TAE from PASTIS would add extra pretraining data and break protocol parity. **Protocol first.**

2. **No clean pretrained checkpoint for our setup**  
   Official U-TAE work uses **PASTIS** (agricultural panoptic SITS). We use **MultiSenGE**: urban LULC, **12 channels** (10 S2 + 2 S1), geographic split, 6/10-class taxonomy. PASTIS weights are not a drop-in match (bands, time, labels, often no S1).

3. **Domain mismatch**  
   PASTIS = crops / panoptic agriculture; MultiSenGE = urban fabric + SAR. Blind transfer can help or hurt; showing TL works would be a **separate ablation**, not the default bake-off vs A4 and concat U-TAE P5.

4. **Thesis / novelty plan intent**  
   Breast paper is **inspiration for the training schedule** (stable head-then-full), not the publishable claim. See `BenchmarkGuide/UTAE_Publishable_Novelty_Plan.md`: do **not** title the work "breast TL for EO." Claim = Task M (fusion), Task H (hierarchy), etc., under the **same MultiSenGE geographic protocol**.

5. **MA-UTAE is separate**  
   `train_ma.py` states MA-UTAE does **not** load concat U-TAE weights (architecture differs). That is intentional ablation isolation, not external TL.

## Architecture note (U-TAE origin)

U-TAE (Sainte Fare Garnot & Landrieu) was developed and evaluated mainly on **PASTIS**. We use the **same architecture family** on MultiSenGE but train **from random init** unless an explicit future ablation says otherwise.

## One sentence for sir / report

> We adopt the breast paper's **probe → head-only → full fine-tune procedure** for training stability; we do **not** initialize from PASTIS or ImageNet because our baselines must match the MultiSenGE paper's **from-scratch** protocol, and PASTIS pretraining is neither fair nor architecturally matched for S1+S2 urban LULC.

## Optional future work (not default)

| Ablation | Purpose |
|----------|---------|
| PASTIS-init U-TAE → MultiSenGE FT | Measure domain-transfer benefit/harm |
| Scratch vs PASTIS-init, same P4→P5 | Isolate TL vs schedule |

Report any such run in `results/RESULTS_BOARD.md` as an extra row, not as replacement for frozen A4 / concat P5 baselines.

## Related files

- Phases: `multisenge_utae/README.md` (Breast-paper phases)
- Breast paper notes: `BenchmarkGuide/` (Singh TCBB 2021 summary in `log.md` 2026 entries)
- Results: `multisenge_utae/results/RESULTS_BOARD.md`
- Init ckpt: `train.py` / `train_ma.py` `--init-ckpt` (same-project P4→P5 only)
