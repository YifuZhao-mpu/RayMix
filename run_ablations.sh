#!/bin/bash
# Codex load-bearing ablations for the RayMix paper (component #2, decomposition #3, volume #4).
# All driven via --options off the single RayMix config, seed0, seq03. Waits for the 3-seed confirm to free GPUs.
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC; export PYTHONPATH=./
CFG=configs/semantic_poss/poss_spunet_raymix_seq03.py

# wait for seed-confirm to finish (frees GPUs 1,2,3,6)
echo "[abl] waiting for seed-confirm to finish..."
while pgrep -f run_L3_seeds.sh >/dev/null; do sleep 60; done
echo "[abl] seed-confirm done; starting ablation queue."

run() {  # name  "extra --options ..."
  local name=$1; shift
  echo "[abl] $name :: $*"
  rm -rf exp/semantic_poss/$name
  CUDA_VISIBLE_DEVICES=1,2,3,6 $PY tools/train.py --config-file $CFG --num-gpus 4 \
    --options seed=0 save_path=exp/semantic_poss/$name "$@" > $LOG/abl_${name}.log 2>&1
}

# ---- component ablation (#2): naive -> +ground_snap -> +validity -> (+ray_merge = full RayMix, already done) ----
run abl_naive  data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False
run abl_snap   data.train.snap_ground=True  data.train.validity=False data.train.use_ray_merge=False
run abl_valid  data.train.snap_ground=True  data.train.validity=True  data.train.use_ray_merge=False
# ---- decomposition (#3): scene-swap only ; instance-transplant only ----
run dec_swaponly data.train.do_transplant=False
run dec_instonly data.train.swap_p=0.0
# ---- volume control (#4): match RayMix accepted volume up toward PolarMix's 3x paste ----
run vol3 data.train.volume_mult=3
echo "[abl] DONE"

$PY - <<'PYEOF'
import re, json
ROOT="/root/project/Pointcept/exp/semantic_poss"
names=["people","rider","car","trunk","plants","sign","pole","trashcan","building","cone","fence","bike","ground"]
WEAK=[3,6,7,9]  # trunk,pole,trashcan,cone
def parse(exp):
    try: txt=open(f"{ROOT}/{exp}/train.log").read()
    except: return None
    bv=re.findall(r"Best mIoU: (0\.\d+)", txt)
    if not bv: return None
    rows=re.findall(r"Class_(\d+)\s*-\s*[\w\-]+\s*Result:\s*iou/accuracy\s*([0-9.]+)/([0-9.]+)", txt)[-13:]
    pc={int(c):round(float(i)*100,1) for c,i,a in rows}
    return dict(best=round(float(bv[-1])*100,2), pc=pc)
rows=[("baseline","poss_spunet_seq03"),("+PolarMix","poss_spunet_polarmix_seq03"),
      ("RayMix-full","poss_spunet_raymix_seq03"),
      ("abl_naive","abl_naive"),("abl_+snap","abl_snap"),("abl_+validity","abl_valid"),
      ("dec_swap-only","dec_swaponly"),("dec_inst-only","dec_instonly"),("vol_x3","vol3")]
print("\n============ RayMix ABLATIONS (seed0, seq03) ============")
print(f"{'arm':16s} {'mIoU':>6s} | trunk pole trash cone")
out={}
for tag,exp in rows:
    r=parse(exp)
    if r is None: print(f"{tag:16s} (pending)"); continue
    wk=" ".join(f"{r['pc'].get(c,float('nan')):4.0f}" for c in WEAK)
    print(f"{tag:16s} {r['best']:6.2f} | {wk}")
    out[tag]=r
json.dump(out, open("/root/Fresh_ARIS8/code/outputs/poss/ablations.json","w"), indent=2, default=str)
print("wrote ablations.json\n========================================================\n")
PYEOF