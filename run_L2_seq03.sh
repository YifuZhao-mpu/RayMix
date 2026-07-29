#!/bin/bash
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC
echo "[L2] seq03 baseline (SpUNet, no PolarMix)..."
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1,2,3,6 bash scripts/train.sh -p $PY -d semantic_poss -c poss_spunet_seq03 -n poss_spunet_seq03 -g 4 > $LOG/L2_base_seq03.log 2>&1
echo "[L2] seq03 +PolarMix..."
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1,2,3,6 bash scripts/train.sh -p $PY -d semantic_poss -c poss_spunet_polarmix_seq03 -n poss_spunet_polarmix_seq03 -g 4 > $LOG/L2_polarmix_seq03.log 2>&1
echo "[L2] DONE"
# machine-print verdict
$PY - <<'PYEOF'
import re
def best(exp):
    try: return float(re.findall(r"Best mIoU: (0\.\d+)", open(f"/root/project/Pointcept/exp/semantic_poss/{exp}/train.log").read())[-1])
    except Exception as e: return None
b=best("poss_spunet_seq03"); p=best("poss_spunet_polarmix_seq03")
print(f"[L2 seq03] SpUNet baseline best mIoU = {b}")
print(f"[L2 seq03] SpUNet+PolarMix best mIoU = {p}")
print(f"PolarMix paper (SPVCNN+PolarMix, seq03) = 0.586")
if b and p:
    print(f"my +PolarMix delta = {(p-b)*100:+.2f}pp ; vs PolarMix-paper 58.6 = {(p-0.586)*100:+.2f}pp")
    print("CLEARS 58.6:" , p>0.586)
PYEOF
