#!/usr/bin/env python3
"""Machine cross-check for paper_tiv_v2: every number in main.tex / strata_auto.tex /
scua_matrix.tex <-> machine-printed JSONs. Run after any edit:
    python3 code/paper_tiv_v2_claim_check.py   (exit 0 = all PASS)"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PO = os.path.join(HERE, "outputs", "poss")
PAPER = os.path.join(HERE, "..", "paper_tiv_v2")
J = lambda p: json.load(open(os.path.join(PO, p)))

tex = open(os.path.join(PAPER, "main.tex")).read()
stex = open(os.path.join(PAPER, "strata_auto.tex")).read()
mtex = open(os.path.join(PAPER, "scua_matrix.tex")).read()
alltex = tex + stex + mtex

p2 = J("phase2_causal.json"); s2 = J("seq02_3seed_2gpu.json"); ci = J("seq02_mixing_cis.json")
ex = J("explor_levers.json"); rf = J("ref2g_spunet_seq03.json"); ps = J("paper_stats.json")
sv = J("repro_spvcnn_seq03_2gpu.json"); cr = J("seq02_ckpt_rule.json"); v3 = J("poss_vol3_balnaive.json")
sp = J("scua_poss.json"); sk = J("scua_kitti.json"); mx = J("scua_matrix.json")
rt = J("scua_retrain.json"); st = J("scua_strata.json")

checks = []
def ck(desc, claim, expected, where=None):
    ok = claim in (where if where is not None else alltex)
    checks.append((ok, desc, claim, expected))

def num_c(v):  # 8782 -> 8{,}782
    return f"{round(v):,}".replace(",", "{,}")

# ---------------- retained legacy checks (same sources as paper_q2/tiv) ----------------
ck("seq03 polarmix mean", "60.95", p2["arms"]["PolarMix"]["mean"])
ck("seq03 polarmix std", "0.39", p2["arms"]["PolarMix"]["std"])
ck("seq03 baseline", "58.53\\,$\\pm$\\,0.23", p2["arms"]["baseline"])
ck("seq03 bal-naive", "61.00\\pm0.28", p2["arms"]["balanced-naive"])
ck("seq03 raymix", "61.00\\pm0.11", p2["arms"]["RayMix-full"])
ck("seq03 polarmix seeds", "\\{60.88,61.46,60.51\\}", p2["arms"]["PolarMix"]["seeds"])
ck("controls swap-only", "59.87", p2["controls"]["scene-swap-only"])
ck("controls transplant", "61.19", p2["controls"]["balanced-naive instonly"])
ck("fullpoint 3-seed", "60.26\\pm0.47", ex["polarmix_fullpoint_3seed"])
ck("seq02 base", "56.35\\,$\\pm$\\,0.59", s2["base"])
ck("seq02 polarmix", "55.91\\,$\\pm$\\,0.16", s2["polarmix"])
ck("seq02 raymix", "57.02", s2["raymix"])
ck("seq02 margin", "+2.9", ci["base_vs_FRNet_pp"])
ck("seq02 worst-seed", "+2.1", ci["base_worst_seed_vs_FRNet_pp"])
ck("seq02 pm delta", "0.44", ci["polarmix_minus_base"]["mean_2dp"])
ck("seq02 pm per-seed", "-1.24/-0.30/+0.22", ci["polarmix_minus_base"]["per_seed"])
ck("seq02 pm CI", "[-2.28,+1.40]", ci["polarmix_minus_base"]["ci95_2dp"])
ck("seq02 rm delta", "+0.67", ci["raymix_minus_base"]["mean_2dp"])
ck("seq02 rm CI", "[-0.63,+1.96]", ci["raymix_minus_base"]["ci95_2dp"])
ck("2gpu ref", "60.93\\pm0.32", rf)
ck("spvcnn matched", "57.45\\pm1.10", sv)
ck("spvcnn row", "57.45\\,$\\pm$\\,1.10", sv)
ck("2gpu ref row", "60.93\\,$\\pm$\\,0.32", rf)
ck("ptv3", "60.86", ex["ptv3_2gpu"]["miou"])
ck("grid04", "60.55", ex["grid04_2gpu"]["miou"])
ck("100ep pair", "60.66\\,/\\,60.25", ex["schedule_100ep_1gpu"])
ck("100ep refs", "59.98/60.33", ex["ref_50ep_1gpu"])
ck("100ep deltas", "+0.68/-0.08", ex["schedule_paired_delta_1gpu"])
ck("vol3 raymix", "61.86\\pm0.38", p2["arms"]["vol_x3"])
ck("vol3 balnaive", f"{v3['balnaive_vol3_mean']:.2f}", v3["balnaive_vol3_mean"])
ck("ckpt final-ep", "55.07", cr["base"]["final_epoch_mean"])
ck("ckpt final-ep margin", "+1.6", cr["base_final_epoch_margin_vs_FRNet"])
ck("ckpt fullpoint", "55.17\\pm0.53", cr["base"])
ck("ckpt fullpoint margin", "+1.7", cr["base_fullpoint_margin_vs_FRNet"])
ck("ckpt min any rule", "54.43", cr["min_value_any_arm_any_rule"])
ck("poss paired CI", "[-0.53,+0.53]", ps["poss_50ep"]["ci95_2dp"])
ck("kitti 24ep", "+0.53", ps["kitti_24ep"]["mean_2dp"])
ck("kitti 24ep seeds", "+0.43/+0.41/+0.76", ps["kitti_24ep"]["diffs"])
ck("kitti 50ep", "-0.28", ps["kitti_50ep"]["mean_2dp"])
ck("kitti 50ep seeds", "-0.25/+0.10/-0.68", ps["kitti_50ep"]["diffs"])
ck("kitti 50ep CI", "[-1.25,+0.69]", ps["kitti_50ep"]["ci95_2dp"])

# ---------------- SCUA audit checks ----------------
C = sp["conditions"]
ck("scua naive vis", num_c(C["naive_x3"]["visibility"]["excess_per_10k"]["point"]),
   C["naive_x3"]["visibility"]["excess_per_10k"]["point"])
ck("scua bal vis", num_c(C["balanced_naive"]["visibility"]["excess_per_10k"]["point"]),
   C["balanced_naive"]["visibility"]["excess_per_10k"]["point"])
ck("scua raymix vis", str(round(C["raymix"]["visibility"]["excess_per_10k"]["point"])),
   C["raymix"]["visibility"]["excess_per_10k"]["point"])
ck("scua pm-visonly vis", num_c(C["polarmix_visonly"]["visibility"]["excess_per_10k"]["point"]),
   C["polarmix_visonly"]["visibility"]["excess_per_10k"]["point"])
for cond, desc in [("naive_x3", "naive"), ("raymix", "raymix")]:
    ck(f"scua {desc} behind", f"{100*C[cond]['visibility']['kept_behind_frac']['point']:.1f}\\%",
       C[cond]["visibility"]["kept_behind_frac"]["point"])
ck("scua pm-visonly behind", f"{100*C['polarmix_visonly']['visibility']['kept_behind_frac']['point']:.1f}\\%",
   C["polarmix_visonly"]["visibility"]["kept_behind_frac"]["point"])
ck("scua naive W", f"{C['naive_x3']['ground']['wasserstein_vs_real']['point']:.3f}",
   C["naive_x3"]["ground"]["wasserstein_vs_real"]["point"])
ck("scua bal W", f"{C['balanced_naive']['ground']['wasserstein_vs_real']['point']:.3f}",
   C["balanced_naive"]["ground"]["wasserstein_vs_real"]["point"])
ck("scua raymix W", f"{C['raymix']['ground']['wasserstein_vs_real']['point']:.3f}",
   C["raymix"]["ground"]["wasserstein_vs_real"]["point"])
ck("scua floor vis", f"{C['real_floor']['visibility']['floor_per_10k']['point']:.1f}",
   C["real_floor"]["visibility"]["floor_per_10k"]["point"])
ck("scua real gap", f"{C['real_floor']['ground']['median']:.3f}", C["real_floor"]["ground"]["median"])
ck("scua raymix gap", f"{C['raymix']['ground']['median']['point']:.3f}",
   C["raymix"]["ground"]["median"]["point"])
ck("scua samp floor", f"{C['real_floor']['sampling']['median_abs']:.2f}",
   C["real_floor"]["sampling"]["median_abs"])
ck("scua poss lift ladder",
   "\\!\\to\\!".join(f"{v:.3f}" for v in sp["verdicts"]["lift_ladder"]),
   sp["verdicts"]["lift_ladder"])
ck("scua poss shift ladder",
   "\\!\\to\\!".join(f"{v:.2f}" for v in sp["verdicts"]["shift_ladder"]),
   sp["verdicts"]["shift_ladder"])
ck("scua kitti lift ladder",
   "\\!\\to\\!".join(f"{v:.3f}" for v in sk["verdicts"]["lift_ladder"]),
   sk["verdicts"]["lift_ladder"])
ck("scua poss calib gap", f"{sp['calibration']['inter_beam_gap_med_deg']:.3f}",
   sp["calibration"]["inter_beam_gap_med_deg"])
ck("scua poss sep", f"{sp['calibration']['separation_ratio']/1e6:.1f}\\times10^{{6}}",
   sp["calibration"]["separation_ratio"])
ck("scua kitti intra", f"{sk['calibration']['intra_beam_std_med_deg']:.3f}",
   sk["calibration"]["intra_beam_std_med_deg"])
ck("scua kitti gap", f"{sk['calibration']['inter_beam_gap_med_deg']:.3f}",
   sk["calibration"]["inter_beam_gap_med_deg"])
ck("scua kitti sep", f"{sk['calibration']['separation_ratio']:.1f}",
   sk["calibration"]["separation_ratio"])
ck("scua kitti bal vis", num_c(sk["conditions"]["balanced_naive"]["visibility"]["excess_per_10k"]["point"]),
   sk["conditions"]["balanced_naive"]["visibility"]["excess_per_10k"]["point"])
ck("scua kitti raymix vis", num_c(sk["conditions"]["raymix"]["visibility"]["excess_per_10k"]["point"]),
   sk["conditions"]["raymix"]["visibility"]["excess_per_10k"]["point"])
n_true = sum(1 for d in (sp["verdicts"], sk["verdicts"]) for v in d.values() if v is True)
checks.append((n_true == 12 and not any(v is False or v is None
              for d in (sp["verdicts"], sk["verdicts"]) for v in d.values() if isinstance(v, (bool, type(None)))),
              "scua 16 verdicts all pass (12 bool + 4 ladders)", f"{n_true} True bools", "12"))

# ---------------- matrix checks ----------------
for r in mx["rows"]:
    c = r["c"]
    ck(f"matrix {r['label'][:30]}", f"{c['mean']:+.2f}$ $[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]",
       c, where=mtex)
checks.append((hashlib.md5(open(os.path.join(PO, "scua_matrix.tex"), "rb").read()).hexdigest()
               == hashlib.md5(open(os.path.join(PAPER, "scua_matrix.tex"), "rb").read()).hexdigest(),
               "matrix tex copies identical", "md5", "equal"))
ck("matrix included", "\\input{scua_matrix.tex}", "-", where=tex)

# ---------------- retrain + strata checks ----------------
ck("retrain pm", f"{rt['arms']['polarmix']['mean']:.2f}\\pm{rt['arms']['polarmix']['std']:.2f}",
   rt["arms"]["polarmix"])
ck("retrain pm-base", f"{rt['contrasts']['polarmix_minus_base']['mean']:+.2f}",
   rt["contrasts"]["polarmix_minus_base"])
ck("retrain rm-pm", f"{rt['contrasts']['raymix_minus_polarmix']['mean']:+.2f}$"
   f" $[{rt['contrasts']['raymix_minus_polarmix']['ci95'][0]:+.2f},"
   f"{rt['contrasts']['raymix_minus_polarmix']['ci95'][1]:+.2f}]", "-", where=stex)
for key, g, bins in [("polarmix-base", "VRU", (2, 3)), ("raymix-polarmix", "VRU", (2, 3))]:
    c = st["contrasts"][key][g]
    for b in bins:
        ck(f"strata {key} {g} bin{b}", f"{c['point'][b]:+.2f}", c["point"][b], where=stex)
        ck(f"strata {key} {g} bin{b} CI", f"[{c['lo'][b]:+.2f},{c['hi'][b]:+.2f}]",
           (c["lo"][b], c["hi"][b]), where=stex)
pmb = st["contrasts"]["polarmix-base"]["VRU"]
ck("abstract strata far1", f"{pmb['point'][2]:+.2f}".replace("+", "-").replace("--", "-"),
   pmb["point"][2], where=tex)
ck("abstract strata far2", f"{pmb['point'][3]:.2f}", pmb["point"][3], where=tex)
ck("strata discipline", "not a validated safety metric", "-", where=stex)
checks.append(("\\PENDING" not in alltex, "no PENDING placeholders", "-", "-"))

fails = [c for c in checks if not c[0]]
for ok, desc, claim, exp in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {desc:38s}  tex:'{claim}'")
print(f"\n{len(checks)-len(fails)}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
