#!/bin/bash
# ============================================================================
# SPVCNN+PolarMix matched-codebase repro, 2-GPU (2026-07-04).
# Pairs with ref2g_spunet_seq03.json (SpUNet+PolarMix 2-GPU: 60.93±0.32) — same
# recipe/protocol/GPU config, only the backbone differs. Prior reported number
# for SPVCNN+PolarMix (seq03): 58.6 (PolarMix, NeurIPS'22).
#   * mean ~60-61 => prior 58.6 was under-tuned; the beat is RECIPE-driven.
#   * mean ~58.6  => the backbone matters; our margin partly backbone.
# GPU constraint: only 4,5 (user 2026-07-03). NCCL: socket transport only.
# ============================================================================
set -u
PC=/root/project/Pointcept
PY=${PY:-/root/miniconda3/envs/pointcept/bin/python}
LOG=/root/Fresh_ARIS8/outputs
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5
cd "$PC"; export PYTHONPATH=./

CFG=configs/semantic_poss/poss_spvcnn_polarmix_seq03.py
$PY -c "import torchsparse; print('[spvcnn2g] torchsparse', torchsparse.__version__)" || { echo "[spvcnn2g] FATAL: torchsparse missing"; exit 1; }
$PY -c "import importlib.util as u; s=u.spec_from_file_location('c','$CFG'); m=u.module_from_spec(s); s.loader.exec_module(m); print('[spvcnn2g] config OK:', m.model['backbone']['type'])" \
  || { echo "[spvcnn2g] FATAL: config import failed"; exit 1; }

for s in 0 1 2; do
  name="spvcnn2g_polarmix_seq03_s${s}"
  echo "[spvcnn2g] $(date '+%F %T') START $name"
  rm -rf exp/semantic_poss/$name
  $PY tools/train.py --config-file $CFG --num-gpus 2 \
    --options seed=$s save_path=exp/semantic_poss/$name > "$LOG/${name}.log" 2>&1
  rc=$?
  echo "[spvcnn2g] $(date '+%F %T') END   $name rc=$rc"
  [ $rc -ne 0 ] && { echo "[spvcnn2g] FATAL: $name failed rc=$rc — aborting."; exit $rc; }
done

$PY - <<'PYEOF'
import re, json, statistics as st
ROOT = "/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try:
        t = open(f"{ROOT}/{e}/train.log").read()
        return round(float(re.findall(r"Best mIoU: (0\.\d+)", t)[-1]) * 100, 2)
    except Exception:
        return None
v = [best(f"spvcnn2g_polarmix_seq03_s{s}") for s in (0, 1, 2)]
vv = [x for x in v if x is not None]
ref = json.load(open("/root/Fresh_ARIS8/code/outputs/poss/ref2g_spunet_seq03.json"))
res = dict(arch="SPVCNN+PolarMix (our recipe, 2-GPU)", seeds=v,
           mean=round(st.mean(vv), 2) if vv else None,
           std=round(st.pstdev(vv), 2) if len(vv) > 1 else 0.0,
           prior_reported=58.6,
           matched_pair_spunet_2gpu=ref,
           note=("Matched-codebase/matched-config backbone pair: compare to SpUNet+PolarMix 2-GPU "
                 f"{ref['mean']}±{ref['std']}. mean ~60-61 => recipe-driven beat; ~58.6 => backbone matters."))
json.dump(res, open("/root/Fresh_ARIS8/code/outputs/poss/repro_spvcnn_seq03_2gpu.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PYEOF
echo "[spvcnn2g] wrote code/outputs/poss/repro_spvcnn_seq03_2gpu.json"
echo "[spvcnn2g] ALL DONE $(date '+%F %T')"
