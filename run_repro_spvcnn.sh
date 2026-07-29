#!/bin/bash
# ============================================================================
# MATCHED-CODEBASE reproduction of the prior best on SemanticPOSS seq03:
#   SPVCNN + PolarMix (reported 58.6 mIoU, PolarMix NeurIPS'22), run under OUR recipe, 3 seeds.
# Converts "we beat a reported NUMBER (60.95 SpUNet vs 58.6)" into "we beat the SAME METHOD under
# matched conditions": same loss/opt/schedule/voxel/aug; only the backbone is SPVCNN.
#   * mean ~60-61  => the +2.4 beat is RECIPE-driven; prior SPVCNN+PolarMix 58.6 was under-tuned. (our thesis)
#   * mean ~58.6   => the SpUNet backbone itself matters.
#
# SINGLE-GPU (--num-gpus 1) to avoid the multi-GPU NCCL "illegal memory access" on this box; jobs run
# one-per-GPU across $GPUS.
#
# Usage:   GPUS=4,5 bash code/run_repro_spvcnn.sh
# ============================================================================
set -u
PC=/root/project/Pointcept
PY=${PY:-/root/miniconda3/envs/pointcept/bin/python}
GPUS=${GPUS:-4,5}
LOG=/root/Fresh_ARIS8/outputs
cd "$PC"; export PYTHONPATH=./
CFG=configs/semantic_poss/poss_spvcnn_polarmix_seq03.py
[ -f "$CFG" ] || { echo "MISSING $CFG — create the SPVCNN config first."; exit 1; }
$PY -c "import importlib.util as u; s=u.spec_from_file_location('c','$CFG'); m=u.module_from_spec(s); s.loader.exec_module(m); print('[repro-spvcnn] config OK:', m.model['backbone']['type'])" \
  || { echo "config failed to import — fix before running."; exit 1; }

IFS=',' read -ra GPU_ARR <<< "$GPUS"
declare -A GPU_PID
wait_free_gpu() {
  while true; do
    for g in "${GPU_ARR[@]}"; do
      local pid=${GPU_PID[$g]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "$g"; return; fi
    done
    sleep 20
  done
}

for s in 0 1 2; do
  name="repro_spvcnn_polarmix_seq03_s${s}"
  g=$(wait_free_gpu)
  echo "[repro-spvcnn] launch $name on GPU $g"
  rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=$g $PY tools/train.py --config-file $CFG --num-gpus 1 \
    --options seed=$s save_path=exp/semantic_poss/$name > "$LOG/${name}.log" 2>&1 &
  GPU_PID[$g]=$!
done
echo "[repro-spvcnn] all jobs queued; waiting..."
wait
echo "[repro-spvcnn] all runs done; aggregating."

$PY - <<'PYEOF'
import re, json, statistics as st
ROOT = "/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try:
        return round(float(re.findall(r"Best mIoU: (0\.\d+)", open(f"{ROOT}/{e}/train.log").read())[-1]) * 100, 2)
    except Exception:
        return None
v = [best(f"repro_spvcnn_polarmix_seq03_s{s}") for s in (0, 1, 2)]
vv = [x for x in v if x is not None]
res = dict(arch="SPVCNN+PolarMix (our recipe)", seeds=v,
           mean=round(st.mean(vv), 2) if vv else None,
           std=round(st.pstdev(vv), 2) if len(vv) > 1 else 0.0,
           prior_reported=58.6, ours_spunet_polarmix=60.95,
           note=("mean ~60-61 => beat is recipe-driven (prior 58.6 under-tuned), matched-condition win; "
                 "mean ~58.6 => SpUNet backbone matters."))
json.dump(res, open("/root/Fresh_ARIS8/code/outputs/poss/repro_spvcnn_seq03.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PYEOF
echo "[repro-spvcnn] wrote code/outputs/poss/repro_spvcnn_seq03.json"
