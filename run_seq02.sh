#!/bin/bash
# seq02 robustness (Codex appendix nice-to-have): same configs, eval_seq=2 (FRNet protocol), 1 seed, 3 arms.
# Runs LAST, after the KITTI chain frees GPUs. Checks the RayMix finding isn't seq03-specific.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
echo "[seq02] waiting for seed/ablation/kitti chains to finish..."
while pgrep -f "run_L3_seeds.sh|run_ablations.sh|run_kitti_sanity.sh" >/dev/null; do sleep 120; done
echo "[seq02] all prior chains done; running seq02 arms."

run() {  # name cfg
  local name=$1 cfg=$2
  echo "[seq02] $name"
  rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file configs/semantic_poss/${cfg}.py --num-gpus 4 \
    --options seed=0 data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2 \
    save_path=exp/semantic_poss/$name > $LOG/seq02_${name}.log 2>&1
}
run seq02_base     poss_spunet_seq03
run seq02_polarmix poss_spunet_polarmix_seq03
run seq02_raymix   poss_spunet_raymix_seq03
echo "[seq02] DONE"
$PY - <<'PYEOF'
import re, json
ROOT="/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try: return round(float(re.findall(r"Best mIoU: (0\.\d+)", open(f"{ROOT}/{e}/train.log").read())[-1])*100,2)
    except: return None
b,p,r=best("seq02_base"),best("seq02_polarmix"),best("seq02_raymix")
print(f"\n===== seq02 robustness (FRNet protocol, seed0) =====\nbaseline {b} | PolarMix {p} | RayMix {r}")
if p and r: print(f"RayMix-PolarMix={r-p:+.2f}pp (expect ~tie, as seq03)")
json.dump(dict(baseline=b,polarmix=p,raymix=r), open("/root/Fresh_ARIS8/code/outputs/poss/seq02.json","w"), indent=2)
print("wrote seq02.json\n===================================================\n")
PYEOF