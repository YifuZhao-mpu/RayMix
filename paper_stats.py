"""Paired 95% CIs for the RayMix vs balanced-naive comparisons (kill-argument fix, 2026-06-16).
Reads ONLY machine-printed per-seed JSONs; reframes the negative result as 'failure to demonstrate
a robust converged gain' with an explicit smallest-effect-of-interest (SESOI). n=3 paired seeds.
t-based 95% CI (df=2, t*=4.302653) is the honest interval for n=3 (no bootstrap with 3 points)."""
import json, math
from decimal import Decimal, ROUND_HALF_UP
O = "/root/Fresh_ARIS8/code/outputs/poss"
T95_DF2 = 4.302653  # two-sided 95% t critical, df=2
SESOI = 0.5         # mIoU: improvements below this are practically negligible vs observed seed std

def r2(x):  # explicit round-half-up to 2dp (paper display precision; avoids float-repr truncation of e.g. 0.045)
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def ci(diffs):
    n = len(diffs); m = sum(diffs)/n
    var = sum((d-m)**2 for d in diffs)/(n-1)
    sd = math.sqrt(var); sem = sd/math.sqrt(n)
    half = T95_DF2*sem
    lo, hi = m-half, m+half
    return dict(diffs=[round(d,3) for d in diffs], mean=round(m,3), std=round(sd,3),
                sem=round(sem,3), ci95=[round(lo,3), round(hi,3)],
                mean_2dp=r2(m), ci95_2dp=[r2(lo), r2(hi)],  # <- values the paper reports verbatim
                ci_includes_0=(lo <= 0 <= hi))

# POSS converged (50ep, 3-seed): RayMix-full - balanced-naive, seed-aligned
ph2 = json.load(open(f"{O}/phase2_causal.json"))
bn = ph2["arms"]["balanced-naive"]["seeds"]; rm = ph2["arms"]["RayMix-full"]["seeds"]
poss = ci([r-b for r,b in zip(rm,bn)])

# KITTI 24ep (3-seed): raymix_full - balanced_naive
kp = json.load(open(f"{O}/kitti_perseed.json"))["arms_24ep"]
bn24 = kp["balanced_naive"]["loop_best_per_seed"]; rm24 = kp["raymix_full"]["loop_best_per_seed"]
k24 = ci([r-b for r,b in zip(rm24,bn24)])

# KITTI 50ep (3-seed): already paired
k50 = ci(json.load(open(f"{O}/kitti_50ep_seeds.json"))["paired_delta_loop"])

out = dict(method="paired t-based 95% CI, n=3 seeds, df=2, t*=4.302653", sesoi_miou=SESOI,
           poss_50ep=poss, kitti_24ep=k24, kitti_50ep=k50)
json.dump(out, open(f"{O}/paper_stats.json","w"), indent=1)
def line(name,c):
    print(f"{name:14s} mean {c['mean']:+.2f}  95%CI [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}]  "
          f"includes0={c['ci_includes_0']}  diffs={c['diffs']}")
print("SESOI =", SESOI, "mIoU")
line("POSS 50ep", poss); line("KITTI 24ep", k24); line("KITTI 50ep", k50)
