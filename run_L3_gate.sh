#!/bin/bash
set -u
PC=/root/project/Pointcept; PY=/root/miniconda3/envs/pointcept/bin/python; LOG=/root/Fresh_ARIS8/outputs
cd $PC
echo "[L3] RayMix seq03 (SpUNet + ray-consistent transplantation), seed0, GPUs 1,2,3,6..."
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1,2,3,6 bash scripts/train.sh -p $PY -d semantic_poss -c poss_spunet_raymix_seq03 -n poss_spunet_raymix_seq03 -g 4 > $LOG/L3_raymix_seq03.log 2>&1
echo "[L3] DONE"
# machine-print verdict from raw train.logs (NEVER hand-typed)
$PY - <<'PYEOF'
import re, json
ROOT="/root/project/Pointcept/exp/semantic_poss"
names=["people","rider","car","trunk","plants","traffic-sign","pole","trashcan","building","cone-stone","fence","bike","ground"]
WEAK=[3,5,6,7,9]  # trunk,sign,pole,trashcan,cone
def parse(exp):
    p=f"{ROOT}/{exp}/train.log"
    try: txt=open(p).read()
    except: return None
    bv=re.findall(r"Best mIoU: (0\.\d+)", txt)
    if not bv: return None
    rows=re.findall(r"Class_(\d+)\s*-\s*[\w\-]+\s*Result:\s*iou/accuracy\s*([0-9.]+)/([0-9.]+)", txt)[-13:]
    pc={int(c): round(float(i)*100,1) for c,i,a in rows}
    return dict(best_val=float(bv[-1]), per_class=pc)
base=parse("poss_spunet_seq03"); pm=parse("poss_spunet_polarmix_seq03"); rm=parse("poss_spunet_raymix_seq03")
print("\n================ L3 GATE VERDICT (seed0, seq03) ================")
print(f"{'':18s} {'best-val':>9s}  | weak-class IoU: " + " ".join(f"{names[c][:5]:>6s}" for c in WEAK))
for tag,r in [("baseline",base),("+PolarMix",pm),("+RayMix(ours)",rm)]:
    if r is None: print(f"{tag:18s} (missing)"); continue
    wk=" ".join(f"{r['per_class'].get(c,float('nan')):6.1f}" for c in WEAK)
    print(f"{tag:18s} {r['best_val']*100:8.2f}   | {wk}")
if rm and pm and base:
    d_pm=(rm['best_val']-pm['best_val'])*100; d_base=(rm['best_val']-base['best_val'])*100
    weak_gain=sum(rm['per_class'].get(c,0)-pm['per_class'].get(c,0) for c in WEAK)
    print(f"\nRayMix vs PolarMix(60.88) = {d_pm:+.2f}pp ; vs baseline = {d_base:+.2f}pp ; vs published 58.6 = {(rm['best_val']-0.586)*100:+.2f}pp")
    print(f"weak-class total IoU gain (RayMix - PolarMix) over {WEAK} = {weak_gain:+.1f} IoU-pts")
    gate = rm['best_val']>pm['best_val'] and weak_gain>0
    print(f"GATE PASS (beat 60.88 AND weak-class net positive): {gate}")
    out=dict(baseline=base,polarmix=pm,raymix=rm,delta_vs_polarmix_pp=round(d_pm,2),
             delta_vs_baseline_pp=round(d_base,2),weak_gain_iou=round(weak_gain,1),gate_pass=bool(gate))
    open("/root/Fresh_ARIS8/code/outputs/poss/L3_gate_seq03.json","w").write(json.dumps(out,indent=2,default=str))
    print("wrote /root/Fresh_ARIS8/code/outputs/poss/L3_gate_seq03.json")
print("================================================================\n")
PYEOF