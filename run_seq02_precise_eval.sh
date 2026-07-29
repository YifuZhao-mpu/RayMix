#!/bin/bash
# ============================================================================
# seq02 checkpoint-rule robustness (2026-07-04): full-point (per-point, whole
# scan) test.py evaluation of the 9 seq02g2 best-val checkpoints, plus
# final-epoch val extraction from the training logs. Bounds the best-val
# checkpoint-selection advantage in Table 2 against priors that may report
# final-epoch numbers. GPUs 4,5 (user constraint), 1-GPU per eval, 2-way parallel.
# ============================================================================
set -u
PC=/root/project/Pointcept
PY=${PY:-/root/miniconda3/envs/pointcept/bin/python}
LOG=/root/Fresh_ARIS8/outputs
cd "$PC"; export PYTHONPATH=./

cfg_for() { case $1 in
  base)     echo poss_spunet_seq03;;
  polarmix) echo poss_spunet_polarmix_seq03;;
  raymix)   echo poss_spunet_raymix_seq03;;
esac; }

GPUS=(4 5); i=0; pids=()
for a in base polarmix raymix; do
  for s in 0 1 2; do
    g=${GPUS[$((i % 2))]}; name="seq02g2_${a}_s${s}"
    echo "[precise] $name on GPU $g"
    CUDA_VISIBLE_DEVICES=$g $PY tools/test.py \
      --config-file configs/semantic_poss/$(cfg_for $a).py --num-gpus 1 \
      --options save_path=exp/semantic_poss/$name \
      weight=exp/semantic_poss/$name/model/model_best.pth \
      data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2 \
      > "$LOG/${name}_precise.log" 2>&1 &
    pids+=($!); i=$((i+1))
    # keep at most 2 concurrent
    if [ ${#pids[@]} -ge 2 ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
  done
done
wait
echo "[precise] all evals done; aggregating (+ final-epoch from train logs)."

$PY - <<'PYEOF'
import re, json, statistics as st
LOG = "/root/Fresh_ARIS8/outputs"
out = {"note": ("seq02 checkpoint-rule robustness: best-val (headline) vs final-epoch (grid val, last epoch "
                "in train log) vs full-point test.py eval of the best-val checkpoint. FRNet prior = 53.5."),
       "FRNet_prior": 53.5}
for a in ("base", "polarmix", "raymix"):
    best, fin, prec = [], [], []
    for s in (0, 1, 2):
        t = open(f"{LOG}/seq02g2_{a}_s{s}.log").read()
        best.append(round(float(re.findall(r"Best mIoU: (0\.\d+)", t)[-1]) * 100, 2))
        fin.append(round(float(re.findall(r"Val result: mIoU/mAcc/allAcc (0\.\d+)", t)[-1]) * 100, 2))
        p = open(f"{LOG}/seq02g2_{a}_s{s}_precise.log").read()
        frames = set(re.findall(r"Test: (\d\d)_", p))
        assert frames == {"02"}, f"precise eval ran on wrong sequence(s): {frames}"
        prec.append(round(float(re.findall(r"Val result: mIoU/mAcc/allAcc (0\.\d+)", p)[-1]) * 100, 2))
    out[a] = {"best_val": best, "best_val_mean": round(st.mean(best), 2),
              "final_epoch": fin, "final_epoch_mean": round(st.mean(fin), 2),
              "fullpoint_best_ckpt": prec, "fullpoint_mean": round(st.mean(prec), 2),
              "fullpoint_pstd": round(st.pstdev(prec), 2)}
out["base_final_epoch_margin_vs_FRNet"] = round(out["base"]["final_epoch_mean"] - 53.5, 2)
out["base_fullpoint_margin_vs_FRNet"] = round(out["base"]["fullpoint_mean"] - 53.5, 2)
out["min_value_any_arm_any_rule"] = min(min(out[a][k] for k in ("best_val", "final_epoch", "fullpoint_best_ckpt"))
                                        for a in ("base", "polarmix", "raymix"))
json.dump(out, open("/root/Fresh_ARIS8/code/outputs/poss/seq02_ckpt_rule.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
echo "[precise] wrote code/outputs/poss/seq02_ckpt_rule.json"
echo "[precise] ALL DONE $(date '+%F %T')"
