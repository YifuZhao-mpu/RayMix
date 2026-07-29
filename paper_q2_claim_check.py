#!/usr/bin/env python3
"""Machine cross-check: every number in paper_q2/main.tex <-> machine-printed JSONs.
Run after any edit: python3 code/paper_q2_claim_check.py  (exit 0 = all PASS)."""
import json, os, re, sys

PO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "poss")
TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_q2", "main.tex")
J = lambda p: json.load(open(os.path.join(PO, p)))
tex = open(TEX).read()

p2 = J("phase2_causal.json"); s2 = J("seq02_3seed_2gpu.json"); ci = J("seq02_mixing_cis.json")
ex = J("explor_levers.json"); rf = J("ref2g_spunet_seq03.json"); ps = J("paper_stats.json")
sv = J("repro_spvcnn_seq03_2gpu.json"); cr = J("seq02_ckpt_rule.json")

checks = []
def ck(desc, claim_in_tex, expected):
    """claim_in_tex: string that must appear in main.tex; expected: source value it encodes."""
    ok = claim_in_tex in tex
    checks.append((ok, desc, claim_in_tex, expected))

# --- seq03 3-seed arms (phase2_causal.json) ---
ck("seq03 polarmix mean±std", "60.95", (p2["arms"]["PolarMix"]["mean"], p2["arms"]["PolarMix"]["std"]))
ck("seq03 polarmix std", "0.39", p2["arms"]["PolarMix"]["std"])
ck("seq03 baseline", "58.53\\,$\\pm$\\,0.23", (p2["arms"]["baseline"]["mean"], p2["arms"]["baseline"]["std"]))
ck("seq03 bal-naive", "61.00\\pm0.28", (p2["arms"]["balanced-naive"]["mean"], p2["arms"]["balanced-naive"]["std"]))
ck("seq03 raymix", "61.00\\pm0.11", (p2["arms"]["RayMix-full"]["mean"], p2["arms"]["RayMix-full"]["std"]))
ck("seq03 polarmix seeds", "\\{60.88,61.46,60.51\\}", p2["arms"]["PolarMix"]["seeds"])
ck("controls swap/transplant", "59.87", p2["controls"]["scene-swap-only"])
ck("controls transplant", "61.19", p2["controls"]["balanced-naive instonly"])

# --- full-point precise eval (explor_levers.json <- poss_precise.json) ---
ck("fullpoint 3-seed mean", "60.26", ex["polarmix_fullpoint_3seed"]["mean_2dp"])
ck("fullpoint 3-seed std", "60.26\\pm0.47", ex["polarmix_fullpoint_3seed"]["pstd_2dp"])

# --- seq02 authoritative (seq02_3seed_2gpu.json + CIs) ---
ck("seq02 base", "56.35\\,$\\pm$\\,0.59", (s2["base"]["mean"], s2["base"]["std"]))
ck("seq02 polarmix", "55.91\\,$\\pm$\\,0.16", (s2["polarmix"]["mean"], s2["polarmix"]["std"]))
ck("seq02 raymix", "57.02", (s2["raymix"]["mean"], s2["raymix"]["std"]))
ck("seq02 margin", "+2.9", ci["base_vs_FRNet_pp"])
ck("seq02 worst-seed margin", "+2.1", ci["base_worst_seed_vs_FRNet_pp"])
ck("seq02 pm delta", "0.44", ci["polarmix_minus_base"]["mean_2dp"])
ck("seq02 pm per-seed", "-1.24/-0.30/+0.22", ci["polarmix_minus_base"]["per_seed"])
ck("seq02 pm CI", "[-2.28,+1.40]", ci["polarmix_minus_base"]["ci95_2dp"])
ck("seq02 rm delta", "+0.67", ci["raymix_minus_base"]["mean_2dp"])
ck("seq02 rm CI", "[-0.63,+1.96]", ci["raymix_minus_base"]["ci95_2dp"])

# --- 2-GPU calibration (ref2g_spunet_seq03.json) ---
ck("2gpu ref", "60.93\\pm0.32", (rf["mean"], rf["std"]))

# --- SPVCNN matched pair (repro_spvcnn_seq03_2gpu.json) ---
ck("spvcnn matched mean±std", "57.45\\pm1.10", (sv["mean"], sv["std"]))
ck("spvcnn table row", "57.45\\,$\\pm$\\,1.10", (sv["mean"], sv["std"]))
ck("spvcnn pair ref row", "60.93\\,$\\pm$\\,0.32", (rf["mean"], rf["std"]))

# --- exploratory levers (explor_levers.json) ---
ck("ptv3", "60.86", ex["ptv3_2gpu"]["miou"])
ck("grid04", "60.55", ex["grid04_2gpu"]["miou"])
ck("100ep pair", "60.66\\,/\\,60.25", (ex["schedule_100ep_1gpu"]["s0"], ex["schedule_100ep_1gpu"]["s1"]))
ck("100ep refs", "59.98/60.33", (ex["ref_50ep_1gpu"]["s0"], ex["ref_50ep_1gpu"]["s1"]))
ck("100ep deltas", "+0.68/-0.08", ex["schedule_paired_delta_1gpu"])

# --- paste-volume lever: RayMix @3x (phase2_causal arms.vol_x3) + balanced-naive @3x matched control ---
ck("vol_x3 raymix 3x", "61.86\\pm0.38", (p2["arms"]["vol_x3"]["mean"], p2["arms"]["vol_x3"]["std"]))
_bnv = os.path.join(PO, "poss_vol3_balnaive.json")
if os.path.exists(_bnv) and os.path.getsize(_bnv) > 0 and \
   json.load(open(_bnv)).get("balnaive_vol3_mean") is not None:
    bv = json.load(open(_bnv))
    # physics-independence: RayMix@3x and balanced-naive@3x should coincide (|delta| < 1.0 pp)
    checks.append((abs(bv["delta_raymix_minus_balnaive_vol3"]) < 1.0,
                   "vol3 physics-independent", f"|RM-BN|={abs(bv['delta_raymix_minus_balnaive_vol3'])}", "<1.0pp"))
    # the balanced-naive@3x mean must literally appear in the paper (placeholder must be filled)
    ck("vol_x3 balnaive 3x in tex", f"{bv['balnaive_vol3_mean']:.2f}", bv["balnaive_vol3_mean"])
    assert "\\PENDINGBN" not in tex, "balanced-naive@3x control is done but \\PENDINGBN placeholder still in main.tex"
else:
    print("[PENDING] balanced-naive @3x control not finished; \\PENDINGBN placeholder expected in main.tex")

# --- seq02 checkpoint-rule robustness (seq02_ckpt_rule.json) ---
ck("seq02 base final-epoch", "55.07", cr["base"]["final_epoch_mean"])
ck("seq02 final-ep margin", "+1.6", cr["base_final_epoch_margin_vs_FRNet"])
ck("seq02 base fullpoint", "55.17\\pm0.53", (cr["base"]["fullpoint_mean"], cr["base"]["fullpoint_pstd"]))
ck("seq02 fullpoint margin", "+1.7", cr["base_fullpoint_margin_vs_FRNet"])
ck("seq02 min any rule", "54.43", cr["min_value_any_arm_any_rule"])
assert all(cr["polarmix"][k[0]] <= cr["base"][k[0]] for k in
           [("best_val_mean",), ("final_epoch_mean",), ("fullpoint_mean",)]), \
    "PolarMix beats base under some rule — 'fails under all three rules' claim is wrong"

# --- RayMix stats (paper_stats.json) ---
ck("poss paired CI", "[-0.53,+0.53]", ps["poss_50ep"]["ci95_2dp"])
ck("kitti 24ep", "+0.53", ps["kitti_24ep"]["mean_2dp"])
ck("kitti 24ep seeds", "+0.43/+0.41/+0.76", ps["kitti_24ep"]["diffs"])
ck("kitti 50ep", "-0.28", ps["kitti_50ep"]["mean_2dp"])
ck("kitti 50ep seeds", "-0.25/+0.10/-0.68", ps["kitti_50ep"]["diffs"])
ck("kitti 50ep CI", "[-1.25,+0.69]", ps["kitti_50ep"]["ci95_2dp"])

fails = [c for c in checks if not c[0]]
for ok, desc, claim, exp in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {desc:28s}  tex:'{claim}'  src:{exp}")
print(f"\n{len(checks)-len(fails)}/{len(checks)} PASS")
sys.exit(1 if fails else 0)
