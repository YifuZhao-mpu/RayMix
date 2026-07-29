"""Central 4-panel figure for the RayMix paper. Reads ONLY from machine-printed result JSONs / raw logs
(no hand-typed numbers). Panels: (a) ray double-returns removed; (b) ground-clearance vs real; (c) honest
deltas RayMix - balanced-naive (POSS 3-seed, KITTI 24ep, KITTI 50ep); (d) KITTI 50ep val curves."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "/root/Fresh_ARIS8/code/outputs/poss"
pv  = json.load(open(f"{O}/phys_validity.json"))
pvk = json.load(open(f"{O}/phys_validity_kitti.json"))
ks  = json.load(open(f"{O}/kitti_seeds.json"))
ph2 = json.load(open(f"{O}/phase2_causal.json"))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# (a) double-returns added per scan (PolarMix vs RayMix), POSS + KITTI
dr_pm = [pv["double_returns"]["polarmix"] if "polarmix" in pv["double_returns"] else pv["double_returns"].get("polarmix_mean"),
         pvk["double_returns"]["polarmix"]]
dr_rm = [pv["double_returns"].get("raymix", pv["double_returns"].get("raymix_mean")),
         pvk["double_returns"]["raymix"]]
x = np.arange(2); w = 0.35
ax[0,0].bar(x-w/2, dr_pm, w, label="PolarMix (naive)", color="#d6604d")
ax[0,0].bar(x+w/2, np.maximum(dr_rm,0), w, label="RayMix (ours)", color="#4393c3")
ax[0,0].set_xticks(x); ax[0,0].set_xticklabels(["POSS (40-beam)","KITTI (64-beam)"])
ax[0,0].set_ylabel("ray double-returns added / scan")
ax[0,0].set_title("(a) Physical artifacts removed"); ax[0,0].legend()
for i,(p,r) in enumerate(zip(dr_pm,dr_rm)):
    ax[0,0].text(i-w/2, p, f"{p:.0f}", ha="center", va="bottom", fontsize=8)
    ax[0,0].text(i+w/2, max(r,0), "$\\leq$0", ha="center", va="bottom", fontsize=8)

# (b) ground-clearance (m): real vs PolarMix vs RayMix, POSS + KITTI
def gc(d):
    g=d["ground_float"]
    return [g.get("real",g.get("real_median")), g.get("polarmix",g.get("polarmix_median")), g.get("raymix",g.get("raymix_median"))]
poss_gc, kitti_gc = gc(pv), gc(pvk)
x2=np.arange(2); w2=0.27
for j,(lab,col) in enumerate([("REAL","#1a9850"),("PolarMix","#d6604d"),("RayMix","#4393c3")]):
    ax[0,1].bar(x2+(j-1)*w2, [poss_gc[j],kitti_gc[j]], w2, label=lab, color=col)
ax[0,1].set_xticks(x2); ax[0,1].set_xticklabels(["POSS","KITTI"])
ax[0,1].set_ylabel("ground-clearance |base - ground| (m)")
ax[0,1].set_title("(b) Ground support: RayMix close to real"); ax[0,1].legend()

# (c) HONEST converged result: RayMix - balanced-naive (no converged-accuracy benefit; 24ep KITTI gain washes out)
k50 = json.load(open(f"{O}/kitti_50ep_seeds.json"))  # 3-seed converged (killarg-v2)
poss_delta = ph2["arms"]["RayMix-full"]["mean"] - ph2["arms"]["balanced-naive"]["mean"]  # 3-seed, converged
deltas=[poss_delta, ks["delta_miou"], k50["delta_mean_loop"]]
labs=["POSS 50ep\n(3-seed)","KITTI 24ep\n(3-seed)","KITTI 50ep\n(3-seed)"]
cols=["#999999","#f4a582","#4393c3"]
bars=ax[1,0].bar(labs, deltas, color=cols)
ax[1,0].axhline(0, color="k", lw=0.8)
ax[1,0].set_ylabel("RayMix-full − balanced-naive  (mIoU)")
ax[1,0].set_title("(c) No converged-accuracy benefit; 24ep KITTI gain washes out")
for b,v in zip(bars,deltas):
    ax[1,0].text(b.get_x()+b.get_width()/2, v, f"{v:+.2f}", ha="center",
                 va="bottom" if v>=0 else "top", fontsize=10, fontweight="bold")

# (d) KITTI 50ep val curves (per-epoch evaluator evals ONLY; excludes the final test.py precise eval)
import re
R="/root/project/Pointcept/exp/semantic_kitti"
def curve(e):
    txt = open(f"{R}/{e}/train.log").read()
    return [float(v)*100 for v in re.findall(r"evaluator\.py line 187 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)]
bnc, rmc = curve("kitti_balnaive_50ep"), curve("kitti_raymix_50ep")
ep=np.arange(1,len(bnc)+1)
ax[1,1].plot(ep, bnc, "-o", ms=3, label="balanced-naive", color="#999999")
ax[1,1].plot(ep, rmc, "-o", ms=3, label="RayMix-full", color="#4393c3")
ax[1,1].set_xlabel("epoch (KITTI 64-beam, 50-epoch schedule)"); ax[1,1].set_ylabel("val mIoU")
ax[1,1].set_ylim(60, 71); ax[1,1].legend(loc="lower right")
ax[1,1].set_title("(d) 50ep schedule: curves cross; balanced-naive finishes ahead")

plt.tight_layout()
plt.savefig("/root/Fresh_ARIS8/paper/central_figure.pdf", bbox_inches="tight")
plt.savefig("/root/Fresh_ARIS8/paper/central_figure.png", dpi=150, bbox_inches="tight")
print("saved central_figure.pdf/.png")
print(f"panel(c): POSS-3seed {deltas[0]:+.2f}, KITTI-24ep {deltas[1]:+.2f}, KITTI-50ep {deltas[2]:+.2f}")
print(f"panel(d): KITTI 50ep converged val mIoU  balanced-naive {bnc[-1]:.1f}  RayMix {rmc[-1]:.1f}")
