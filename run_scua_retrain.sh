#!/bin/bash
# SCUA strata-rescoring retrain: 3 arms x 3 seeds, ALL single-GPU (per-device batch 12) so the
# nine runs are internally BN-consistent; published 1-GPU 50ep PolarMix refs (59.98/60.33) anchor
# the config externally. Wave A: 6 jobs on GPUs 1-6. Wave B: 3 jobs on GPUs 1-3.
# POSS checkpoints were not migrated to this machine; these runs also regenerate the release assets.
set -u
PC=/workspace/project/Pointcept
PY=/workspace/miniconda3/envs/pointcept/bin/python
LOG=/workspace/Fresh_ARIS8/outputs
mkdir -p "$LOG"
cd "$PC"; export PYTHONPATH=.

BASE=configs/semantic_poss/poss_spunet_seq03.py
PM=configs/semantic_poss/poss_spunet_polarmix_seq03.py
RM=configs/semantic_poss/poss_spunet_raymix_seq03.py

run() { # run NAME CONFIG GPU SEED [extra options...]
  local name=$1 cfg=$2 gpu=$3 seed=$4; shift 4
  rm -rf "exp/semantic_poss/$name"
  echo "[scua-retrain] launch $name (gpu $gpu, seed $seed)"
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train.py --config-file "$cfg" --num-gpus 1 \
    --options save_path="exp/semantic_poss/$name" seed="$seed" "$@" \
    > "$LOG/scua_${name}.log" 2>&1
  echo "[scua-retrain] done $name (exit $?)"
}

# ---- wave A: 6 concurrent single-GPU jobs ----
run scua_base_s0 "$BASE" 1 0 & run scua_base_s1 "$BASE" 2 1 & run scua_base_s2 "$BASE" 3 2 &
run scua_pm_s0   "$PM"   4 0 & run scua_pm_s1   "$PM"   5 1 & run scua_pm_s2   "$PM"   6 2 &
wait
echo "[scua-retrain] wave A complete"

# ---- wave B: 3 concurrent RayMix jobs ----
run scua_rm_s0 "$RM" 1 0 & run scua_rm_s1 "$RM" 2 1 & run scua_rm_s2 "$RM" 3 2 &
wait
echo "[scua-retrain] wave B complete"

for d in scua_base_s0 scua_base_s1 scua_base_s2 scua_pm_s0 scua_pm_s1 scua_pm_s2 \
         scua_rm_s0 scua_rm_s1 scua_rm_s2; do
  b=$(grep -oE "Best validation mIoU updated to: [0-9.]+" "exp/semantic_poss/$d/train.log" \
      2>/dev/null | tail -1)
  echo "RESULT $d :: ${b:-MISSING}"
done
echo "[scua-retrain] ALL DONE"
