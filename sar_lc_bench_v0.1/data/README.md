# Data layout (not shipped as large rasters in Git)

SAR-LC-Bench **does not** redistribute the full AI4LCC MultiSenGE / MultiSenNA TIFF archives.
Obtain rasters and labels from the official AI4LCC sources, then place files as below.

## Recommended layout (public clone or Release unpack)

```text
sar_lc_bench_v0.1/
  data/
    ge/
      ai4lcc_test.jsonl          # 2497 rows (copy from research tree)
      s1_test_bench/             # 2497 VH GeoTIFFs for those patch_ids only
    na/                          # optional transfer
      multisenna_bench.jsonl
      s1_na_bench/               # or point to full NA s1 pack
```

## Where to get the pieces

| Artifact | How |
|----------|-----|
| Labels / patch metadata | AI4LCC MultiSenGE / MultiSenNA releases |
| Full S1 `.tif` stacks | AI4LCC `s1.tgz` (large; use sir PC / server if needed) |
| Frozen GE test JSONL | Copy `ai4lcc_test.jsonl` — see `BENCH_MANIFEST_v0.1.json` SHA256 |
| Compact GE TIFF pack | Build once with `python -m lulcdial.pack_bench_s1` against the test JSONL |

## Verify the GE bench file

```bash
# Linux / macOS / Git Bash
sha256sum data/ge/ai4lcc_test.jsonl
# expect: 90b8dedc4c905aeb004f11522363928d0905535803c9f3cf6c14528be2325b30
```

For public download of the **small** TIFF pack (~2497 files), prefer **Zenodo** or a **GitHub Release** (not the git history).
