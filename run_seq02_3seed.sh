#!/bin/bash
# ============================================================================
# seq02 (FRNet protocol) 3-SEED — authoritative replacement for the single-seed
# seq02 number in Table 2, and the fix for the verified PROVENANCE BUG:
#   * Paper's Table 2 "58.52 = recipe+PolarMix" is actually NO-MIXING recipe
#     (raw log outputs/poss_seq02_train.log: SemanticPOSSDataset, poss_scratch_seq02, Best mIoU 0.5852).
#   * On seq02 PolarMix appears to HURT (Jun-5 single-seed sweep: base 58.14 / polarmix 56.26 / raymix 57.50).
# Runs base / +PolarMix / +RayMix at seeds {0,1,2} on eval_seq=2 (train={0,1,3,4,5}, val={2}).
#
# SINGLE-GPU (--num-gpus 1) to avoid the multi-GPU NCCL "illegal memory access" seen on the 6,7 pair.
# Jobs are parallelised one-per-GPU across $GPUS (POSS fits easily in one 96GB GPU).
#
# Usage:   GPUS=4,5 bash code/run_seq02_3seed.sh        # 2-way parallel on GPUs 4 and 5
#          GPUS=4   bash code/run_seq02_3seed.sh        # strictly sequential on GPU 4
# ============================================================================
set -u
PC=/root/project/Pointcept
PY=${PY:-/root/miniconda3/envs/pointcept/bin/python}
GPUS=${GPUS:-4,5}
LOG=/root/Fresh_ARIS8/outputs
cd "$PC"; export PYTHONPATH=./
IFS=',' read -ra GPU_ARR <<< "$GPUS"
declare -A GPU_PID

wait_free_gpu() {  # echo the index of a free GPU (blocks until one frees)
  while true; do
    for g in "${GPU_ARR[@]}"; do
      local pid=${GPU_PID[$g]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "$g"; return; fi
    done
    sleep 20
  done
}

launch() {  # arm cfg seed
  local arm=$1 cfg=$2 s=$3 name="seq02_${1}_s${3}"
  local g; g=$(wait_free_gpu)
  echo "[seq02-3seed] launch $name on GPU $g  (cfg=$cfg eval_seq=2 seed=$s)"
  rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=$g $PY tools/train.py \
    --config-file configs/semantic_poss/${cfg}.py --num-gpus 1 \
    --options seed=$s data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2 \
    save_path=exp/semantic_poss/$name > "$LOG/${name}.log" 2>&1 &
  GPU_PID[$g]=$!
}

for s in 0 1 2; do
  launch base     poss_spunet_seq03          $s
  launch polarmix poss_spunet_polarmix_seq03 $s
  launch raymix   poss_spunet_raymix_seq03   $s
done
echo "[seq02-3seed] all jobs queued; waiting for completion..."
wait
echo "[seq02-3seed] all runs done; aggregating."

$PY - <<'PYEOF'
import re, json, statistics as st
ROOT = "/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try:
        t = open(f"{ROOT}/{e}/train.log").read()
        return round(float(re.findall(r"Best mIoU: (0\.\d+)", t)[-1]) * 100, 2)
    except Exception:
        return None
arms = {a: [best(f"seq02_{a}_s{s}") for s in (0, 1, 2)] for a in ("base", "polarmix", "raymix")}
res = {}
for a, v in arms.items():
    vv = [x for x in v if x is not None]
    res[a] = {"seeds": v,
              "mean": round(st.mean(vv), 2) if vv else None,
              "std": round(st.pstdev(vv), 2) if len(vv) > 1 else 0.0}
pb = [arms["polarmix"][i] - arms["base"][i] for i in range(3)
      if arms["polarmix"][i] is not None and arms["base"][i] is not None]
res["polarmix_minus_base"] = {"per_seed": [round(x, 2) for x in pb],
                              "mean": round(st.mean(pb), 2) if pb else None}
res["FRNet_prior"] = 53.5
res["note"] = ("Authoritative seq02. Expect base (no-mixing) to be the headline beat vs FRNet 53.5, and "
               "PolarMix <= base (hurts on seq02). Relabel Table 2: '58.52 +PolarMix' -> 'recipe (no mixing)' "
               "+ add a PolarMix row.")
json.dump(res, open("/root/Fresh_ARIS8/code/outputs/poss/seq02_3seed.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PYEOF
echo "[seq02-3seed] wrote code/outputs/poss/seq02_3seed.json"
