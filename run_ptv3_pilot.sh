#!/bin/bash
# PT-v3 second-backbone PILOT (RayMix paper, closes kill-argument P_6 scope). 2026-06-17.
# Tests whether the "physical validity yields no robust converged gain" result generalizes from
# sparse-voxel SpUNet to a TRANSFORMER backbone (PT-v3) on SemanticKITTI 64-beam, 50ep (converged regime).
# GPU 5 (user-approved 2026-06-17; GPU1 was occupied by another job). batch_size=8 (smoke-tested, ~18h/run, ~40GB).
# Data pipeline + RayMix dataset IDENTICAL to SpUNet arms; only backbone+recipe differ (PT-v3's own recipe,
# shared across both PT-v3 arms so the RayMix-vs-balnaive DELTA is clean). NEVER fabricate; machine-print.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
CFG=configs/semantic_kitti/kitti_ptv3_raymix.py
NOPHYS="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"

prun(){ local name=$1 seed=$2; shift 2; rm -rf exp/semantic_kitti/$name
  CUDA_VISIBLE_DEVICES=5 $PY tools/train.py --config-file $CFG --num-gpus 1 \
   --options batch_size=8 epoch=50 eval_epoch=50 seed=$seed save_path=exp/semantic_kitti/$name "$@" > $LOG/${name}.log 2>&1; }

echo "[ptv3-pilot] $(date) seed0 balnaive (no-phys)"; prun ptv3_balnaive_50ep_s0 0 $NOPHYS
echo "[ptv3-pilot] $(date) seed0 raymix";             prun ptv3_raymix_50ep_s0 0
echo "[ptv3-pilot] $(date) DONE; machine-printing pilot verdict"

$PY - <<'PYEOF'
import re, json
def parse(path):
    try: txt = open(path).read()
    except FileNotFoundError: return (None, None, 0)
    loop = [float(v)*100 for v in re.findall(r"evaluator\.py line 187 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)]
    tp = re.findall(r"test\.py line 340 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)
    return (round(max(loop),2) if loop else None,
            round(float(tp[-1])*100,2) if tp else None, len(loop))
RK="/root/project/Pointcept/exp/semantic_kitti"
b = parse(f"{RK}/ptv3_balnaive_50ep_s0/train.log"); r = parse(f"{RK}/ptv3_raymix_50ep_s0/train.log")
out = {"backbone":"PT-v3m1 (transformer)","dataset":"SemanticKITTI 64-beam, 50ep, seed0","batch_size":8,"gpu":5,
       "balnaive":{"loop_best":b[0],"test_precise":b[1],"n_evals":b[2]},
       "raymix":{"loop_best":r[0],"test_precise":r[1],"n_evals":r[2]}}
if b[0] is not None and r[0] is not None:
    d = round(r[0]-b[0],2); out["delta_loop"]=d
    if b[1] is not None and r[1] is not None: out["delta_test_precise"]=round(r[1]-b[1],2)
    # pilot interpretation (single seed): negative thesis REPLICATES if RayMix does not clearly win
    out["pilot_read"] = ("REPLICATES negative thesis (RayMix does not clearly beat balnaive at convergence) -> expand to 3 seeds" if d <= 0.3
                         else "SURPRISE: RayMix wins on PT-v3 seed0 (delta>+0.3) -> do NOT silently expand; surface to user/Codex")
json.dump(out, open("/root/Fresh_ARIS8/code/outputs/poss/ptv3_pilot.json","w"), indent=1)
print("[ptv3-pilot] PT-v3 KITTI 50ep seed0: balnaive", b[0], "raymix", r[0],
      "| delta", out.get("delta_loop"), "|", out.get("pilot_read"))
PYEOF
echo "[ptv3-pilot] $(date) ALL DONE"
