# Sync 10c MA / S1 / S2 results + plots from PARAM (no .pt weights).
# Run in PowerShell from E:\MTP\earth2 (needs your SSH key / agent).
$ErrorActionPreference = "Stop"
$hostName = "rihak_iitp@paramrudra.iitp.ac.in"
$remote = "~/MTP/earth2"
$root = "E:\MTP\earth2"
Set-Location $root

function Ensure-Dir([string]$p) {
  New-Item -ItemType Directory -Force -Path $p | Out-Null
}

# --- result JSONs / md ---
$dirs = @(
  "multisenge_utae/results/ma_utae/ma_c10_gated_head_v0",
  "multisenge_utae/results/ma_utae/ma_c10_gated_full_v0",
  "multisenge_utae/results/ma_utae/ma_c10_concat_head_v0",
  "multisenge_utae/results/ma_utae/ma_c10_concat_full_v0",
  "multisenge_utae/results/concat_utae/run_c10_s1_head_v0",
  "multisenge_utae/results/concat_utae/run_c10_s2_head_v0",
  "multisenge_utae/results/concat_utae/run_c10_s1_full_v0",
  "multisenge_utae/results/concat_utae/run_c10_s2_full_v0"
)
foreach ($d in $dirs) {
  Ensure-Dir $d
  scp -r "${hostName}:${remote}/${d}/." ".\$($d -replace '/','\')\" 2>$null
}

# --- training plots / history from checkpoints ---
$ckpts = @(
  "ma_c10_gated_head_v0",
  "ma_c10_gated_full_v0",
  "ma_c10_concat_head_v0",
  "ma_c10_concat_full_v0",
  "run_c10_s1_head_v0",
  "run_c10_s2_head_v0",
  "run_c10_s1_full_v0",
  "run_c10_s2_full_v0"
)
foreach ($c in $ckpts) {
  $isMa = $c.StartsWith("ma_")
  $dest = if ($isMa) { "multisenge_utae\results\ma_utae\$c" } else { "multisenge_utae\results\concat_utae\$c" }
  Ensure-Dir $dest
  scp "${hostName}:${remote}/multisenge_utae/checkpoints/${c}/history_plot.png" ".\$dest\" 2>$null
  scp "${hostName}:${remote}/multisenge_utae/checkpoints/${c}/history.json" ".\$dest\" 2>$null
  scp "${hostName}:${remote}/multisenge_utae/checkpoints/${c}/best_metrics.json" ".\$dest\" 2>$null
}

Write-Host "Done. Check run_c10_s{1,2}_full_v0 and ma_c10_concat_full_v0 under results/."
