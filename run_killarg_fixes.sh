#!/bin/bash
# Kill-argument fixes (Codex binding decision C, 2026-06-11):
#  1) KITTI 50ep paired seeds 1,2 for balanced-naive + RayMix-full  -> closes P_2 (3-seed 50ep total incl. existing seed0)
#  2) POSS balanced-naive x3-volume, seeds 0,1,2                    -> closes P_5 (realism x volume interaction)
# HARD CONSTRAINT: GPU 1 ONLY (user 2026-06-07). batch_size is global in Pointcept -> identical hyperparams on 1 GPU.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
KCFG=configs/semantic_kitti/kitti_spunet_raymix_fast.py
PCFG=configs/semantic_poss/poss_spunet_raymix_seq03.py
NOPHYS="data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False"

krun(){ local name=$1 seed=$2; shift 2; rm -rf exp/semantic_kitti/$name
  CUDA_VISIBLE_DEVICES=1 $PY tools/train.py --config-file $KCFG --num-gpus 1 \
   --options epoch=50 eval_epoch=50 seed=$seed save_path=exp/semantic_kitti/$name "$@" > $LOG/${name}.log 2>&1; }
prun(){ local name=$1 seed=$2; shift 2; rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=1 $PY tools/train.py --config-file $PCFG --num-gpus 1 \
   --options seed=$seed save_path=exp/semantic_poss/$name "$@" > $LOG/${name}.log 2>&1; }

echo "[killarg] KITTI 50ep seed1: balnaive"; krun kitti_balnaive_50ep_s1 1 $NOPHYS
echo "[killarg] KITTI 50ep seed1: raymix";   krun kitti_raymix_50ep_s1 1
echo "[killarg] KITTI 50ep seed2: balnaive"; krun kitti_balnaive_50ep_s2 2 $NOPHYS
echo "[killarg] KITTI 50ep seed2: raymix";   krun kitti_raymix_50ep_s2 2
echo "[killarg] POSS balnaive x3vol seed0";  prun poss_balnaive_vol3_s0 0 $NOPHYS data.train.volume_mult=3
echo "[killarg] POSS balnaive x3vol seed1";  prun poss_balnaive_vol3_s1 1 $NOPHYS data.train.volume_mult=3
echo "[killarg] POSS balnaive x3vol seed2";  prun poss_balnaive_vol3_s2 2 $NOPHYS data.train.volume_mult=3
echo "[killarg] training DONE; machine-printing verdict JSONs"

$PY - <<'PYEOF'
import re, json, numpy as np
def parse(path):
    txt = open(path).read()
    loop = [float(v)*100 for v in re.findall(r"evaluator\.py line 187 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)]
    tp = re.findall(r"test\.py line 340 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)
    return (round(max(loop),2) if loop else None,
            round(float(tp[-1])*100,2) if tp else None, len(loop))
RK="/root/project/Pointcept/exp/semantic_kitti"; RP="/root/project/Pointcept/exp/semantic_poss"
# KITTI 3-seed 50ep (seed0 = existing 3-GPU runs; seeds 1,2 = new 1-GPU runs; note in JSON)
k = {"note":"seed0 trained on 3 GPUs (pre-constraint), seeds 1,2 on GPU1 only; batch_size global=12 identical; within-pair comparisons share hardware",
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
k["paired_delta_test_precise"]=[round(r-b,2) for r,b in zip(rt,bt)]
k["delta_mean_test_precise"]=round(float(np.mean(rt))-float(np.mean(bt)),4)
json.dump(k, open("/root/Fresh_ARIS8/code/outputs/poss/kitti_50ep_seeds.json","w"), indent=1)
print("[killarg] KITTI 50ep 3-seed: balnaive", bl, "raymix", rl, "paired", k["paired_delta_loop"], "mean_delta", k["delta_mean_loop"])
# POSS balanced-naive x3vol 3-seed vs existing RayMix vol_x3 (phase2_causal.json)
p={"arms":{}}
vals=[]; tps=[]
for s in (0,1,2):
    lb,tp,ne = parse(f"{RP}/poss_balnaive_vol3_s{s}/train.log")
    p["arms"][f"s{s}"]={"loop_best":lb,"test_precise":tp,"n_evals":ne}; vals.append(lb); tps.append(tp)
p["balnaive_vol3_mean"]=round(float(np.mean(vals)),4); p["balnaive_vol3_std"]=round(float(np.std(vals)),4)
p["balnaive_vol3_test_precise_mean"]=round(float(np.mean(tps)),4)
ph2=json.load(open("/root/Fresh_ARIS8/code/outputs/poss/phase2_causal.json"))
rmv=ph2["arms"]["vol_x3"]
p["raymix_vol3_mean"]=rmv["mean"]; p["raymix_vol3_std"]=rmv["std"]; p["raymix_vol3_seeds"]=rmv["seeds"]
p["delta_raymix_vol3_minus_balnaive_vol3"]=round(rmv["mean"]-p["balnaive_vol3_mean"],4)
json.dump(p, open("/root/Fresh_ARIS8/code/outputs/poss/poss_vol3_balnaive.json","w"), indent=1)
print("[killarg] POSS balnaive x3vol:", vals, "mean", p["balnaive_vol3_mean"], "+-", p["balnaive_vol3_std"],
      "| RayMix x3vol mean", rmv["mean"], "| delta(RM-BN)", p["delta_raymix_vol3_minus_balnaive_vol3"])
PYEOF
echo "[killarg] ALL DONE"
