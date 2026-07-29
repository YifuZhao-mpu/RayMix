#!/bin/bash
# Phase-2 (Codex decorrelation-paper must-add): multi-seed CAUSAL arms to prove balanced-naive == RayMix-full on
# mIoU+small-classes (so physical realism is decorrelated from accuracy). Runs after KITTI frees GPUs.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
RM=configs/semantic_poss/poss_spunet_raymix_seq03.py
BASE=configs/semantic_poss/poss_spunet_seq03.py
NAIVE="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"
echo "[p2] waiting for KITTI chain to finish..."
while pgrep -f "run_kitti_sanity.sh" >/dev/null; do sleep 120; done
echo "[p2] KITTI done; running phase-2 causal arms."
run() { local name=$1 cfg=$2; shift 2
  echo "[p2] $name :: $*"; rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file $cfg --num-gpus 4 \
    --options save_path=exp/semantic_poss/$name "$@" > $LOG/p2_${name}.log 2>&1; }
# 3-seed completion (baseline/abl_naive/vol3 only have s0; PolarMix & RayMix already 3-seed)
run baseline_s1 $BASE seed=1
run baseline_s2 $BASE seed=2
run balnaive_s1 $RM seed=1 $NAIVE
run balnaive_s2 $RM seed=2 $NAIVE
run vol3_s1 $RM seed=1 data.train.volume_mult=3
run vol3_s2 $RM seed=2 data.train.volume_mult=3
# 1-seed causal controls
run balnaive_instonly $RM seed=0 $NAIVE data.train.swap_p=0.0
run swaponly $RM seed=0 data.train.do_transplant=False
# optional robustness: seq02
run seq02_base $BASE seed=0 data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2
run seq02_polarmix configs/semantic_poss/poss_spunet_polarmix_seq03.py seed=0 data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2
run seq02_raymix $RM seed=0 data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2
echo "[p2] DONE"
$PY - <<'PYEOF'
import re, json, numpy as np
ROOT="/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try: return round(float(re.findall(r"Best mIoU: (0\.\d+)", open(f"{ROOT}/{e}/train.log").read())[-1])*100,2)
    except: return None
def ms(*es):
    v=[best(e) for e in es]; v=[x for x in v if x is not None]
    return (round(float(np.mean(v)),2), round(float(np.std(v)),2), v) if v else (None,None,[])
arms={
 "baseline":        ms("poss_spunet_seq03","baseline_s1","baseline_s2"),
 "PolarMix":        ms("poss_spunet_polarmix_seq03","poss_spunet_polarmix_seq03_s1","poss_spunet_polarmix_seq03_s2"),
 "balanced-naive":  ms("abl_naive","balnaive_s1","balnaive_s2"),
 "RayMix-full":     ms("poss_spunet_raymix_seq03","poss_spunet_raymix_seq03_s1","poss_spunet_raymix_seq03_s2"),
 "vol_x3":          ms("vol3","vol3_s1","vol3_s2"),
}
ctrl={"balanced-naive instonly":best("balnaive_instonly"),"scene-swap-only":best("swaponly"),
      "RayMix instonly":best("dec_instonly")}
seq02={"base":best("seq02_base"),"PolarMix":best("seq02_polarmix"),"RayMix":best("seq02_raymix")}
print("\n========== PHASE-2 CAUSAL TABLE (3-seed mIoU mean±std) ==========")
for k,(m,s,v) in arms.items(): print(f"  {k:16s} {m}±{s}  {v}")
print("  -- 1-seed controls:", ctrl)
print("  -- seq02 robustness:", seq02)
bn=arms["balanced-naive"][0]; rm=arms["RayMix-full"][0]
if bn and rm: print(f"\n  KEY: balanced-naive {bn} vs RayMix-full {rm} -> delta {rm-bn:+.2f}pp (decorrelation if |.|<0.5)")
json.dump(dict(arms={k:{'mean':m,'std':s,'seeds':v} for k,(m,s,v) in arms.items()},controls=ctrl,seq02=seq02),
          open("/root/Fresh_ARIS8/code/outputs/poss/phase2_causal.json","w"),indent=2)
print("wrote phase2_causal.json\n================================================================\n")
PYEOF