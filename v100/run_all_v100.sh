#!/bin/bash
# ============================================================================
# V100 (4-GPU) runner for the two pre-submission experiments:
#   (1) seq02 3-seed  (base / +PolarMix / +RayMix)  -> seq02_3seed.json
#   (2) SPVCNN+PolarMix matched-codebase repro, seq03, 3-seed -> repro_spvcnn_seq03.json
# Multi-GPU (--num-gpus 4) so per-GPU BN batch = 12/4 = 3, MATCHING the paper's seq03 4-GPU regime.
# NCCL_P2P_DISABLE=1 is set as a harmless safety net (V100 P2P is usually fine).
#
# EDIT THESE for your V100 box, or pass as env vars:
#   PC   = Pointcept checkout on the V100 box
#   DATA = SemanticPOSS data_root on the V100 box
#   PY   = python in the V100 conda env
#   OUT  = where to write the result JSONs
#   GPUS = the 4 V100 indices ; NG = 4
#
# Usage:  PC=~/Pointcept DATA=~/data/SemanticPOSS PY=$(which python) GPUS=0,1,2,3 bash run_all_v100.sh
# ============================================================================
set -u
PC=${PC:-$HOME/Pointcept}
DATA=${DATA:-$HOME/data/SemanticPOSS}
PY=${PY:-python}
OUT=${OUT:-$PC/q2_results}
GPUS=${GPUS:-0,1,2,3}; NG=${NG:-4}
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONPATH=$PC
mkdir -p "$OUT"; cd "$PC"
DOPT="data.train.data_root=$DATA data.val.data_root=$DATA data.test.data_root=$DATA"

run() {  # name cfg seed extra_opts
  local name=$1 cfg=$2 s=$3 extra=${4:-}
  echo "[v100] $(date +%H:%M:%S) $name  (cfg=$cfg seed=$s ng=$NG)"
  rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=$GPUS $PY tools/train.py --config-file configs/semantic_poss/${cfg}.py --num-gpus $NG \
    --options seed=$s $DOPT $extra save_path=exp/semantic_poss/$name > "$OUT/${name}.log" 2>&1
}

echo "===== (1) seq02 3-seed (FRNet protocol, eval_seq=2) ====="
EVAL2="data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2"
for s in 0 1 2; do
  run seq02_base_s$s     poss_spunet_seq03          $s "$EVAL2"
  run seq02_polarmix_s$s poss_spunet_polarmix_seq03 $s "$EVAL2"
  run seq02_raymix_s$s   poss_spunet_raymix_seq03   $s "$EVAL2"
done

echo "===== (2) SPVCNN+PolarMix repro (seq03) ====="
$PY -c "import importlib.util as u;s=u.spec_from_file_location('c','configs/semantic_poss/poss_spvcnn_polarmix_seq03.py');m=u.module_from_spec(s);s.loader.exec_module(m);print('SPVCNN cfg OK')" || echo "WARN: SPVCNN cfg import failed (torchsparse missing?)"
for s in 0 1 2; do run repro_spvcnn_s$s poss_spvcnn_polarmix_seq03 $s ""; done

echo "===== aggregate ====="
OUT="$OUT" $PY - <<'PYEOF'
import re, json, os, statistics as st
ROOT=os.path.join(os.environ.get("PC", os.path.expanduser("~/Pointcept")), "exp/semantic_poss")
OUT=os.environ["OUT"]
def best(e):
    try: return round(float(re.findall(r"Best mIoU: (0\.\d+)", open(f"{ROOT}/{e}/train.log").read())[-1])*100,2)
    except Exception: return None
# seq02
arms={a:[best(f"seq02_{a}_s{s}") for s in (0,1,2)] for a in ("base","polarmix","raymix")}
seq02={}
for a,v in arms.items():
    vv=[x for x in v if x is not None]
    seq02[a]={"seeds":v,"mean":round(st.mean(vv),2) if vv else None,"std":round(st.pstdev(vv),2) if len(vv)>1 else 0.0}
pb=[arms["polarmix"][i]-arms["base"][i] for i in range(3) if arms["polarmix"][i] is not None and arms["base"][i] is not None]
seq02["polarmix_minus_base"]={"per_seed":[round(x,2) for x in pb],"mean":round(st.mean(pb),2) if pb else None}
seq02["FRNet_prior"]=53.5; seq02["regime"]="4-GPU V100, batch 12 (3/GPU), matches paper seq03 BN regime"
json.dump(seq02, open(f"{OUT}/seq02_3seed.json","w"), indent=2); print("seq02:", json.dumps(seq02))
# spvcnn
v=[best(f"repro_spvcnn_s{s}") for s in (0,1,2)]; vv=[x for x in v if x is not None]
sp=dict(arch="SPVCNN+PolarMix (our recipe, V100 4-GPU)", seeds=v,
        mean=round(st.mean(vv),2) if vv else None, std=round(st.pstdev(vv),2) if len(vv)>1 else 0.0,
        prior_reported=58.6, ours_spunet_polarmix=60.95)
json.dump(sp, open(f"{OUT}/repro_spvcnn_seq03.json","w"), indent=2); print("spvcnn:", json.dumps(sp))
PYEOF
echo "[v100] DONE. JSONs in $OUT/"
