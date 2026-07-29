#!/bin/bash
# ============================================================================
# Q2 paper — authoritative 2-GPU chain (2026-07-03).
# CONSTRAINT: only GPUs 4,5 may be used (user directive 2026-07-03).
#
# WHY 2-GPU: the paper's seq03 headline numbers are 4-GPU (batch 3/GPU). The
# Jun-29 single-GPU seq02 3-seed sweep is NOT comparable: per-GPU BatchNorm
# batch (3 vs 12) shifts POSS results ~2.2pp (4-GPU base s0 58.14 vs 1-GPU
# 56.04, configs otherwise identical — verified by config diff). With only 2
# GPUs, we run 2-GPU DDP (batch 6/GPU, global batch 12): the exact config the
# smoke_nccl_p2p run validated end-to-end on Jun 29. Each table is internally
# consistent (all "Ours" rows share one config); the reproducibility section
# states the per-protocol GPU config.
#
# STEP 1: 2-epoch NCCL smoke on 4,5 (abort chain if multi-GPU is broken)
# STEP 2: seq02 sweep — base/polarmix/raymix x seeds {0,1,2}   -> Table 2
# STEP 3: SpUNet+PolarMix seq03 x seeds {0,1,2} (2-GPU reference for the
#         SPVCNN matched-pair comparison; SPVCNN runs in a separate chain
#         once torchsparse is built)
# All runs sequential on CUDA_VISIBLE_DEVICES=4,5 with NCCL_P2P_DISABLE=1.
# ============================================================================
set -u
PC=/root/project/Pointcept
PY=${PY:-/root/miniconda3/envs/pointcept/bin/python}
LOG=/root/Fresh_ARIS8/outputs
# Minimal-allreduce sweep (2026-07-03): P2P and SHM transports BOTH produce
# "illegal memory access" on this box (RTX PRO 6000 Blackwell, NCCL 2.26.2);
# only the socket transport works => disable both.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5
cd "$PC"; export PYTHONPATH=./

run() {  # name cfg extra_options...
  local name=$1 cfg=$2; shift 2
  echo "[chain2g] $(date '+%F %T') START $name (cfg=$cfg $*)"
  rm -rf exp/semantic_poss/$name
  $PY tools/train.py --config-file configs/semantic_poss/${cfg}.py --num-gpus 2 \
    --options "$@" save_path=exp/semantic_poss/$name > "$LOG/${name}.log" 2>&1
  local rc=$?
  echo "[chain2g] $(date '+%F %T') END   $name rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "[chain2g] FATAL: $name failed (rc=$rc) — aborting chain. See $LOG/${name}.log"
    exit $rc
  fi
}

echo "[chain2g] START $(date '+%F %T')  GPUS=4,5  NCCL_P2P_DISABLE=1"

echo "[chain2g] === STEP 1/3: 2-epoch NCCL smoke ==="
run smoke2g_0703 poss_spunet_seq03 seed=0 epoch=2 eval_epoch=2
grep -q "End Evaluation" exp/semantic_poss/smoke2g_0703/train.log \
  || { echo "[chain2g] FATAL: smoke did not reach evaluation — aborting."; exit 1; }
echo "[chain2g] smoke OK."

echo "[chain2g] === STEP 2/3: seq02 sweep (3 arms x 3 seeds, FRNet protocol) ==="
for s in 0 1 2; do
  run seq02g2_base_s${s}     poss_spunet_seq03          seed=$s data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2
  run seq02g2_polarmix_s${s} poss_spunet_polarmix_seq03 seed=$s data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2
  run seq02g2_raymix_s${s}   poss_spunet_raymix_seq03   seed=$s data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2
done

$PY - <<'PYEOF'
import re, json, statistics as st
ROOT = "/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try:
        t = open(f"{ROOT}/{e}/train.log").read()
        return round(float(re.findall(r"Best mIoU: (0\.\d+)", t)[-1]) * 100, 2)
    except Exception:
        return None
arms = {a: [best(f"seq02g2_{a}_s{s}") for s in (0, 1, 2)] for a in ("base", "polarmix", "raymix")}
res = {}
for a, v in arms.items():
    vv = [x for x in v if x is not None]
    res[a] = {"seeds": v,
              "mean": round(st.mean(vv), 2) if vv else None,
              "std": round(st.pstdev(vv), 2) if len(vv) > 1 else 0.0}
for m in ("polarmix", "raymix"):
    d = [arms[m][i] - arms["base"][i] for i in range(3)
         if arms[m][i] is not None and arms["base"][i] is not None]
    res[f"{m}_minus_base"] = {"per_seed": [round(x, 2) for x in d],
                              "mean": round(st.mean(d), 2) if d else None}
res["FRNet_prior"] = 53.5
res["config"] = "2xGPU DDP (batch 6/GPU, global 12), NCCL_P2P_DISABLE=1, GPUs 4,5"
res["note"] = ("AUTHORITATIVE seq02 (FRNet protocol) 3-seed, internally consistent config. "
               "Replaces both the mislabeled single-seed 58.52 (4-GPU, no-mixing) and the "
               "1-GPU 3-seed sweep (BN-batch confound).")
json.dump(res, open("/root/Fresh_ARIS8/code/outputs/poss/seq02_3seed_2gpu.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PYEOF
echo "[chain2g] wrote code/outputs/poss/seq02_3seed_2gpu.json"

echo "[chain2g] === STEP 3/3: SpUNet+PolarMix seq03 2-GPU reference (3 seeds) ==="
for s in 0 1 2; do
  run ref2g_spunet_polarmix_seq03_s${s} poss_spunet_polarmix_seq03 seed=$s
done

$PY - <<'PYEOF'
import re, json, statistics as st
ROOT = "/root/project/Pointcept/exp/semantic_poss"
def best(e):
    try:
        t = open(f"{ROOT}/{e}/train.log").read()
        return round(float(re.findall(r"Best mIoU: (0\.\d+)", t)[-1]) * 100, 2)
    except Exception:
        return None
v = [best(f"ref2g_spunet_polarmix_seq03_s{s}") for s in (0, 1, 2)]
vv = [x for x in v if x is not None]
res = dict(arch="SpUNet+PolarMix seq03 (2-GPU reference)", seeds=v,
           mean=round(st.mean(vv), 2) if vv else None,
           std=round(st.pstdev(vv), 2) if len(vv) > 1 else 0.0,
           four_gpu_reference=60.95,
           note="2-GPU arm of the SPVCNN-vs-SpUNet matched pair; also calibrates 2-GPU vs 4-GPU config offset.")
json.dump(res, open("/root/Fresh_ARIS8/code/outputs/poss/ref2g_spunet_seq03.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PYEOF
echo "[chain2g] wrote code/outputs/poss/ref2g_spunet_seq03.json"
echo "[chain2g] ALL DONE $(date '+%F %T')"
