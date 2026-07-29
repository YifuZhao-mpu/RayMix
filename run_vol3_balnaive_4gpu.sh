#!/bin/bash
# ARIS re-eval fix (2026-07-13): complete the MATCHED balanced-naive @ 3x paste-volume control that
# closes the vol_x3 gap flagged by the GPT-5.6-Sol ARIS review.
#   - Existing RayMix @ 3x vol (phase2_causal.json -> arms.vol_x3) = 61.86±0.38, trained 4-GPU (run_phase2.sh).
#   - This runs balanced-naive (RayMix pipeline, physics OFF) @ 3x vol at the SAME 4-GPU config, 3 seeds,
#     so RayMix-vol3 vs balnaive-vol3 is a clean matched pair (no BN-batch confound).
# If balnaive-vol3 ~= 61.8 too -> the 3x lift is volume/oversampling, not physical validity (supports thesis).
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
RM=configs/semantic_poss/poss_spunet_raymix_seq03.py
NAIVE="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"
run(){ local name=$1 seed=$2; shift 2; rm -rf exp/semantic_poss/$name
  echo "[vol3bn] $name (seed $seed) starting $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file $RM --num-gpus 4 \
    --options save_path=exp/semantic_poss/$name seed=$seed data.train.volume_mult=3 $NAIVE "$@" \
    > $LOG/${name}.log 2>&1
  echo "[vol3bn] $name done $(date -u +%H:%M:%S) rc=$?"; }

run poss_balnaive_vol3_4g_s0 0
run poss_balnaive_vol3_4g_s1 1
run poss_balnaive_vol3_4g_s2 2
echo "[vol3bn] all training done; machine-printing poss_vol3_balnaive.json"

$PY - <<'PYEOF'
import re, json, numpy as np
RP="/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try: return round(float(re.findall(r"Best mIoU: (0\.\d+)", open(f"{RP}/{e}/train.log").read())[-1])*100,2)
    except Exception: return None
seeds=[f"poss_balnaive_vol3_4g_s{s}" for s in (0,1,2)]
vals=[best(e) for e in seeds]; vals=[v for v in vals if v is not None]
ph2=json.load(open("/root/Fresh_ARIS8/code/outputs/poss/phase2_causal.json"))
rmv=ph2["arms"]["vol_x3"]
out={
 "note":"balanced-naive (RayMix pipeline, physics OFF) @ volume_mult=3, seq03, 4-GPU (CUDA 1,2,3,6) — MATCHED to RayMix vol_x3. Closes P_5 / ARIS-2026-07-13 vol_x3 gap.",
 "config":"4xGPU DDP, batch 3/GPU global 12, RM config + snap_ground/validity/use_ray_merge=False + volume_mult=3",
 "balnaive_vol3_seeds":vals,
 "balnaive_vol3_mean":round(float(np.mean(vals)),2) if vals else None,
 "balnaive_vol3_std":round(float(np.std(vals)),2) if vals else None,
 "raymix_vol3_mean":rmv["mean"], "raymix_vol3_std":rmv["std"], "raymix_vol3_seeds":rmv["seeds"],
 "delta_raymix_minus_balnaive_vol3":round(rmv["mean"]-float(np.mean(vals)),2) if vals else None,
 "vol1_refs":{"balanced-naive":ph2["arms"]["balanced-naive"]["mean"],"RayMix-full":ph2["arms"]["RayMix-full"]["mean"],"PolarMix":ph2["arms"]["PolarMix"]["mean"]},
}
json.dump(out, open("/root/Fresh_ARIS8/code/outputs/poss/poss_vol3_balnaive.json","w"), indent=1)
print("[vol3bn] balnaive_vol3:", vals, "mean", out["balnaive_vol3_mean"], "±", out["balnaive_vol3_std"],
      "| RayMix vol3", rmv["mean"], "| delta(RM-BN)", out["delta_raymix_minus_balnaive_vol3"])
PYEOF
echo "[vol3bn] ALL DONE $(date -u +%H:%M:%S)"
