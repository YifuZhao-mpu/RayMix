#!/bin/bash
# Codex Q3 decisive cheap experiment: 2 more seeds each of PolarMix & RayMix (seed0 already done).
# If RayMix mean delta is within ±0.5pp of PolarMix -> TIE confirmed -> bank honest ~7 paper.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
run() {  # cfg seed
  local cfg=$1 seed=$2; local name=${cfg}_s${seed}
  echo "[seeds] $name ..."
  rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file configs/semantic_poss/${cfg}.py --num-gpus 4 \
     --options seed=$seed save_path=exp/semantic_poss/$name > $LOG/seed_${name}.log 2>&1
}
run poss_spunet_polarmix_seq03 1
run poss_spunet_polarmix_seq03 2
run poss_spunet_raymix_seq03 1
run poss_spunet_raymix_seq03 2
echo "[seeds] DONE"
$PY - <<'PYEOF'
import re, numpy as np, json
ROOT="/root/project/Pointcept/exp/semantic_poss"
def best(name):
    try: return round(float(re.findall(r"Best mIoU: (0\.\d+)", open(f"{ROOT}/{name}/train.log").read())[-1])*100,2)
    except: return None
def collect(cfg):
    vals=[best(cfg)]+[best(f"{cfg}_s{s}") for s in (1,2)]   # seed0 dir has no suffix
    return [v for v in vals if v is not None]
pm=collect("poss_spunet_polarmix_seq03"); rm=collect("poss_spunet_raymix_seq03")
print("\n================ L3 MULTI-SEED (seq03) ================")
print(f"PolarMix seeds {pm} -> mean {np.mean(pm):.2f} std {np.std(pm):.2f}")
print(f"RayMix   seeds {rm} -> mean {np.mean(rm):.2f} std {np.std(rm):.2f}")
if len(pm)>=2 and len(rm)>=2:
    d=np.mean(rm)-np.mean(pm)
    print(f"RayMix - PolarMix mean delta = {d:+.2f}pp ; |delta|<=0.5 (TIE -> bank ~7): {abs(d)<=0.5}")
    json.dump(dict(polarmix=pm,raymix=rm,polarmix_mean=round(float(np.mean(pm)),2),
                   raymix_mean=round(float(np.mean(rm)),2),delta_pp=round(float(d),2),
                   tie=bool(abs(d)<=0.5)),
              open("/root/Fresh_ARIS8/code/outputs/poss/L3_seeds.json","w"), indent=2)
    print("wrote L3_seeds.json")
print("======================================================\n")
PYEOF