#!/bin/bash
# Codex must-add: KITTI 50-epoch paired sanity (seed0) to defuse 24ep-artifact criticism. GPUs 2,3,6.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
RM=configs/semantic_kitti/kitti_spunet_raymix_fast.py
N="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"
run(){ local name=$1; shift; rm -rf exp/semantic_kitti/$name
  CUDA_VISIBLE_DEVICES=2,3,6 $PY tools/train.py --config-file $RM --num-gpus 3 \
   --options epoch=50 eval_epoch=50 seed=0 save_path=exp/semantic_kitti/$name "$@" > $LOG/${name}.log 2>&1; }
echo "[50ep] kitti_balnaive_50ep"; run kitti_balnaive_50ep $N
echo "[50ep] kitti_raymix_50ep";   run kitti_raymix_50ep
echo "[50ep] DONE"
$PY - <<'PYEOF'
import re,json
R="/root/project/Pointcept/exp/semantic_kitti"
def best(e):
 try:return round(float(re.findall(r"Best mIoU: (0\.\d+)",open(f"{R}/{e}/train.log").read())[-1])*100,2)
 except:return None
b,r=best("kitti_balnaive_50ep"),best("kitti_raymix_50ep")
d=round(r-b,2) if b and r else None
print(f"[50ep] balanced-naive {b} vs RayMix-full {r} = {d:+}mIoU ; safe(>=+0.3): {d is not None and d>=0.3}")
json.dump(dict(balnaive=b,raymix=r,delta=d,safe=bool(d is not None and d>=0.3)),open("/root/Fresh_ARIS8/code/outputs/poss/kitti_50ep.json","w"),indent=2)
PYEOF
