#!/bin/bash
# Kill-argument fixes v2 — Codex thread 019ec946 binding decision "PATH B" (2026-06-15):
#  Restart REDUCED subset of the dead bqzxxvcni chain. Drop the 3 POSS balnaive-x3vol runs
#  (P_5 made textually moot by demoting the unmatched vol_x3 claim in main.tex instead).
#  Keep ONLY the 4 KITTI 50ep paired runs -> makes the converged claim a 3-seed result (closes P_2).
# HARD CONSTRAINT: GPU 1 ONLY (user 2026-06-07; Codex reaffirmed). batch_size global=12 -> 1-GPU hyperparam-identical.
# All from scratch (no fragile resume of the ep45 partial); seed0 already homogeneous (same config+invocation).
# NEVER fabricate. Every number machine-printed from raw train.log.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
KCFG=configs/semantic_kitti/kitti_spunet_raymix_fast.py
NOPHYS="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"

krun(){ local name=$1 seed=$2; shift 2; rm -rf exp/semantic_kitti/$name
  CUDA_VISIBLE_DEVICES=1 $PY tools/train.py --config-file $KCFG --num-gpus 1 \
   --options epoch=50 eval_epoch=50 seed=$seed save_path=exp/semantic_kitti/$name "$@" > $LOG/${name}.log 2>&1; }

echo "[killarg-v2] $(date) KITTI 50ep seed1: balnaive"; krun kitti_balnaive_50ep_s1 1 $NOPHYS
echo "[killarg-v2] $(date) KITTI 50ep seed1: raymix";   krun kitti_raymix_50ep_s1 1
echo "[killarg-v2] $(date) KITTI 50ep seed2: balnaive"; krun kitti_balnaive_50ep_s2 2 $NOPHYS
echo "[killarg-v2] $(date) KITTI 50ep seed2: raymix";   krun kitti_raymix_50ep_s2 2
echo "[killarg-v2] $(date) training DONE; machine-printing 3-seed verdict JSON"

$PY - <<'PYEOF'
import re, json, numpy as np
def parse(path):
    try: txt = open(path).read()
    except FileNotFoundError: return (None, None, 0)
    loop = [float(v)*100 for v in re.findall(r"evaluator\.py line 187 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)]
    tp = re.findall(r"test\.py line 340 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)
    return (round(max(loop),2) if loop else None,
            round(float(tp[-1])*100,2) if tp else None, len(loop))
RK="/root/project/Pointcept/exp/semantic_kitti"
# seed0 = existing 3-GPU runs (pre-constraint); seeds 1,2 = new GPU1-only runs. batch_size global=12 identical.
k = {"note":"seed0 trained on 3 GPUs (pre-constraint), seeds 1,2 on GPU1 only; batch_size global=12 identical; "
            "within-seed pairs (balnaive vs raymix) share hardware so paired deltas are clean",
     "go_no_go":"Codex path-B: GO (negative thesis holds) iff mean(RayMix-balnaive) loop_best <= +0.25 OR signs mixed; "
                "NO-GO (rewrite, do not submit) iff RayMix beats balnaive on all 3 seeds AND mean >= +0.30",
     "balnaive":{}, "raymix":{}}
for s,(bn,rm) in {0:("kitti_balnaive_50ep","kitti_raymix_50ep"),
                  1:("kitti_balnaive_50ep_s1","kitti_raymix_50ep_s1"),
                  2:("kitti_balnaive_50ep_s2","kitti_raymix_50ep_s2")}.items():
    b = parse(f"{RK}/{bn}/train.log"); r = parse(f"{RK}/{rm}/train.log")
    k["balnaive"][s]={"loop_best":b[0],"test_precise":b[1],"n_evals":b[2]}
    k["raymix"][s]={"loop_best":r[0],"test_precise":r[1],"n_evals":r[2]}
bl=[k["balnaive"][s]["loop_best"] for s in (0,1,2)]; rl=[k["raymix"][s]["loop_best"] for s in (0,1,2)]
k["paired_delta_loop"]=[round(r-b,2) for r,b in zip(rl,bl)]
k["balnaive_mean"]=round(float(np.mean(bl)),4); k["raymix_mean"]=round(float(np.mean(rl)),4)
k["delta_mean_loop"]=round(k["raymix_mean"]-k["balnaive_mean"],4)
bt=[k["balnaive"][s]["test_precise"] for s in (0,1,2)]; rt=[k["raymix"][s]["test_precise"] for s in (0,1,2)]
if all(x is not None for x in bt+rt):
    k["paired_delta_test_precise"]=[round(r-b,2) for r,b in zip(rt,bt)]
    k["delta_mean_test_precise"]=round(float(np.mean(rt))-float(np.mean(bt)),4)
# verdict
deltas=k["paired_delta_loop"]; mean_d=k["delta_mean_loop"]
all_pos = all(d>0 for d in deltas); mixed = (any(d>0 for d in deltas) and any(d<=0 for d in deltas))
no_go = all_pos and mean_d>=0.30
k["verdict"]= "NO-GO (rewrite; RayMix robustly wins at convergence)" if no_go else \
              ("GO (negative thesis holds)" if (mean_d<=0.25 or mixed) else "BORDERLINE — surface to user/Codex")
json.dump(k, open("/root/Fresh_ARIS8/code/outputs/poss/kitti_50ep_seeds.json","w"), indent=1)
print("[killarg-v2] KITTI 50ep 3-seed loop_best: balnaive", bl, "raymix", rl)
print("[killarg-v2] paired deltas", deltas, "| mean", mean_d, "| VERDICT:", k["verdict"])
PYEOF
echo "[killarg-v2] $(date) ALL DONE"
