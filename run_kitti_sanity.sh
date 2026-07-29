#!/bin/bash
# Cross-sensor SANITY (Codex must-have): SemanticKITTI 64-beam, 1 seed, 24-ep controlled, 3 arms.
# Waits for the POSS seed-confirm + ablation queues to free GPUs 1,2,3,6. Machine-prints the ±1.0 gate.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
echo "[kitti] waiting for POSS queues (seed-confirm + ablations) to finish..."
while pgrep -f "run_L3_seeds.sh|run_ablations.sh" >/dev/null; do sleep 120; done
echo "[kitti] POSS queues done; starting KITTI sanity."

run() {  # name  "extra --options"
  local name=$1; shift
  echo "[kitti] $name :: $*"
  rm -rf exp/semantic_kitti/$name
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file $CFG --num-gpus 4 \
    --options seed=0 save_path=exp/semantic_kitti/$name "$@" > $LOG/kitti_${name}.log 2>&1
}
CFG=configs/semantic_kitti/kitti_spunet_base_fast.py
run kitti_base
CFG=configs/semantic_kitti/kitti_spunet_raymix_fast.py
run kitti_polarmix data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False
run kitti_raymix
echo "[kitti] DONE"

$PY - <<'PYEOF'
import re, json
ROOT="/root/project/Pointcept/exp/semantic_kitti"
names=["car","bicycle","motorcycle","truck","other-veh","person","bicyclist","motorcyclist","road","parking",
       "sidewalk","other-grnd","building","fence","veg","trunk","terrain","pole","sign"]
SMALL=[1,2,5,6,7,15,17,18]  # bicycle,motorcycle,person,bicyclist,motorcyclist,trunk,pole,sign
def parse(exp):
    try: txt=open(f"{ROOT}/{exp}/train.log").read()
    except: return None
    bv=re.findall(r"Best mIoU: (0\.\d+)", txt)
    if not bv: return None
    rows=re.findall(r"Class_(\d+)\s*-\s*[\w\-]+\s*Result:\s*iou/accuracy\s*([0-9.]+)/([0-9.]+)", txt)[-19:]
    pc={int(c):round(float(i)*100,1) for c,i,a in rows}
    return dict(miou=round(float(bv[-1])*100,2), pc=pc)
b=parse("kitti_base"); p=parse("kitti_polarmix"); r=parse("kitti_raymix")
print("\n========= KITTI 64-beam CROSS-SENSOR SANITY (24ep, seed0) =========")
for tag,x in [("baseline",b),("PolarMix-naive",p),("RayMix",r)]:
    if x is None: print(f"{tag:16s} (pending)"); continue
    sm=" ".join(f"{x['pc'].get(c,float('nan')):4.0f}" for c in SMALL)
    print(f"{tag:16s} mIoU {x['miou']:5.2f} | small[bic mc per bcy mcy trk pol sgn]: {sm}")
if p and r:
    d=r['miou']-p['miou']
    nstable=sum(1 for c in SMALL if r['pc'].get(c,0) >= p['pc'].get(c,0)-0.5)
    print(f"\nRayMix - PolarMix(naive) = {d:+.2f} mIoU ; small classes stable/up: {nstable}/{len(SMALL)}")
    print(f"GATE (NOT a disaster = within -1.0 mIoU AND >=2 small stable): {d>=-1.0 and nstable>=2}")
    json.dump(dict(baseline=b,polarmix=p,raymix=r,delta=round(d,2),small_stable=nstable),
              open("/root/Fresh_ARIS8/code/outputs/poss/kitti_sanity.json","w"),indent=2,default=str)
    print("wrote kitti_sanity.json")
print("==================================================================\n")
PYEOF