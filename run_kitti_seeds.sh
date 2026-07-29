#!/bin/bash
# Codex FORK: multi-seed KITTI to test the sensor-density hypothesis (physical components help on 64-beam).
# 4 paired runs (balanced-naive & RayMix-full, seeds 1,2) -> 3-seed paired with seed0. Hard gate; if PASS,
# auto-run supporting arms (baseline s1,s2 + KITTI component ablation). Runs after POSS phase2 frees GPUs.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
RM=configs/semantic_kitti/kitti_spunet_raymix_fast.py
BASE=configs/semantic_kitti/kitti_spunet_base_fast.py
NAIVE="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"
echo "[ks] waiting for POSS phase2 to finish..."
while pgrep -f "run_phase2.sh" >/dev/null; do sleep 120; done
echo "[ks] phase2 done; running KITTI multi-seed fork."
run() { local name=$1 cfg=$2; shift 2
  echo "[ks] $name :: $*"; rm -rf exp/semantic_kitti/$name
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file $cfg --num-gpus 4 \
    --options save_path=exp/semantic_kitti/$name "$@" > $LOG/ks_${name}.log 2>&1; }
# 4 paired seed runs (seed0 already = kitti_polarmix[balanced-naive], kitti_raymix[full])
run kbalnaive_s1 $RM seed=1 $NAIVE
run kraymix_s1   $RM seed=1
run kbalnaive_s2 $RM seed=2 $NAIVE
run kraymix_s2   $RM seed=2
echo "[ks] paired seeds DONE; computing gate."
GATE=$($PY - <<'PYEOF'
import re,json,numpy as np
ROOT="/root/project/Pointcept/exp/semantic_kitti"
SMALL=[1,2,5,6,7,15,17,18]
def parse(e):
    try: txt=open(f"{ROOT}/{e}/train.log").read()
    except: return None
    bv=re.findall(r"Best mIoU: (0\.\d+)",txt)
    if not bv: return None
    rows=re.findall(r"Class_(\d+)\s*-\s*[\w\-]+\s*Result:\s*iou/accuracy\s*([0-9.]+)/([0-9.]+)",txt)[-19:]
    pc={int(c):float(i)*100 for c,i,a in rows}
    return float(bv[-1])*100, pc
def arm(es):
    ms=[parse(e) for e in es]; ms=[m for m in ms if m]
    miou=np.mean([m[0] for m in ms]); sm={c:np.mean([m[1].get(c,0) for m in ms]) for c in SMALL}
    return miou, sm, len(ms)
bn=arm(["kitti_polarmix","kbalnaive_s1","kbalnaive_s2"])
rm=arm(["kitti_raymix","kraymix_s1","kraymix_s2"])
dmiou=rm[0]-bn[0]
small_delta=np.mean([rm[1][c]-bn[1][c] for c in SMALL])
nup=sum(1 for c in SMALL if rm[1][c]>=bn[1][c]-0.1)
out=dict(balanced_naive_miou=round(bn[0],2),raymix_miou=round(rm[0],2),delta_miou=round(dmiou,2),
         small_delta=round(float(small_delta),2),small_up=nup,n_seeds=min(bn[2],rm[2]),
         small_bn={int(c):round(bn[1][c],1) for c in SMALL},small_rm={int(c):round(rm[1][c],1) for c in SMALL})
json.dump(out,open("/root/Fresh_ARIS8/code/outputs/poss/kitti_seeds.json","w"),indent=2)
import sys
print(f"KITTI 3-seed: balanced-naive {bn[0]:.2f} vs RayMix-full {rm[0]:.2f} = {dmiou:+.2f}mIoU; small mean {small_delta:+.2f}; {nup}/8 up",file=sys.stderr)
print("PASS" if (dmiou>=0.30 and small_delta>=1.0 and nup>=6) else "FAIL")
PYEOF
)
echo "[ks] GATE = $GATE"
if [ "$GATE" = "PASS" ]; then
  echo "[ks] gate PASS -> sensor-density hypothesis holds; running supporting arms."
  run kbaseline_s1 $BASE seed=1
  run kbaseline_s2 $BASE seed=2
  run ksnap_s0  $RM seed=0 data.train.validity=False data.train.use_ray_merge=False
  run kvalid_s0 $RM seed=0 data.train.use_ray_merge=False
  echo "[ks] supporting arms DONE."
else
  echo "[ks] gate FAIL -> KITTI ambiguous; fall back to workshop decorrelation/null framing."
fi
echo "[ks] ALL DONE"
cat /root/Fresh_ARIS8/code/outputs/poss/kitti_seeds.json