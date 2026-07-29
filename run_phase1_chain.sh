#!/bin/bash
# Phase 1 autonomous chain: wait for converged baseline -> train converged teacher (same 4 GPUs) -> converged
# upper-bound re-test -> machine-printed verdict. The decisive gate for whether amodal is alive.
set -u
PC=/root/project/Pointcept
EXP=$PC/exp/semantic_kitti
PY=/root/miniconda3/envs/pointcept/bin/python
LOG=/root/Fresh_ARIS8/outputs

echo "[chain] waiting for converged baseline to finish..."
until [ -f $EXP/fresh_amodal_baseline_spunet/model/model_best.pth ] && \
      grep -q "Best mIoU" $EXP/fresh_amodal_baseline_spunet/train.log 2>/dev/null && \
      ! pgrep -f "fresh_amodal_baseline_spunet" >/dev/null 2>&1; do
  sleep 300
done
echo "[chain] baseline done. Best:"; grep "Best mIoU" $EXP/fresh_amodal_baseline_spunet/train.log | tail -1

echo "[chain] training converged TEACHER (multi-sweep static-aware, 4-GPU)..."
cd $PC
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1,4,5 bash scripts/train.sh -p $PY -d semantic_kitti \
  -c fresh_amodal_teacher_spunet -n fresh_amodal_teacher_spunet -g 4 > $LOG/teacher_train.log 2>&1
echo "[chain] teacher done. Best:"; grep "Best mIoU" $EXP/fresh_amodal_teacher_spunet/train.log 2>/dev/null | tail -1

echo "[chain] === CONVERGED UPPER-BOUND RE-TEST ==="
cd /root/Fresh_ARIS8/code
CUDA_VISIBLE_DEVICES=0 $PY converged_upperbound.py 2>&1 | grep -vE "spconv|Warning|warn"
echo "[chain] Phase 1 complete. Verdict in outputs/ub_converged/upperbound.json"
