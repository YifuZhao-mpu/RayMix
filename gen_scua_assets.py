#!/usr/bin/env python3
"""SCUA paper assets, machine-generated from audited JSONs (no hand-typed numbers):
  1. outputs/poss/scua_matrix.json  — every matched contrast (paired t CI, n=3, df=2) + which of
     P/E/Q/H it moves; values cross-asserted against previously published JSONs where they exist.
  2. outputs/poss/scua_matrix.tex   — the matched identification matrix as a LaTeX table body.
  3. outputs/poss/fig_scua.pdf      — two-panel figure: (left) the three acquisition-consistency
     residuals per condition with scan-level bootstrap CIs vs the real-scan floor;
     (right) forest plot of the matched converged utility contrasts vs the SESOI band.
Publication style follows gen_result_figs.py (serif, vector, direct labels, no in-figure titles).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PO = os.path.join(HERE, "outputs", "poss")
J = lambda p: json.load(open(os.path.join(PO, p)))
T95 = 4.302653  # t*, df=2

p2 = J("phase2_causal.json")
ps = J("paper_stats.json")
ci2 = J("seq02_mixing_cis.json")
v3 = J("poss_vol3_balnaive.json")
sp = J("scua_poss.json")
sk = J("scua_kitti.json")
s2 = J("seq02_3seed_2gpu.json")


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d.mean()
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=round(float(m), 2), ci=[round(float(m - T95 * sem), 2),
                                             round(float(m + T95 * sem), 2)],
                per_seed=[round(float(x), 2) for x in d])


arm = p2["arms"]
rows = [
    dict(label="PolarMix on vs. off (sequence 03)", moves="E+Q+P", protocol="seq03",
         c=paired(arm["PolarMix"]["seeds"], arm["baseline"]["seeds"]),
         infer="everything moves at once: the composite augmentation effect"),
    dict(label="PolarMix on vs. off (sequence 02)", moves="E+Q+P", protocol="seq02",
         c=dict(mean=ci2["polarmix_minus_base"]["mean_2dp"],
                ci=ci2["polarmix_minus_base"]["ci95_2dp"],
                per_seed=ci2["polarmix_minus_base"]["per_seed"]),
         infer="protocol sensitivity: the same composite effect, other protocol"),
    dict(label="RayMix vs. balanced-naive (sequence 03, 1x)", moves="P", protocol="seq03",
         c=dict(mean=ps["poss_50ep"]["mean_2dp"], ci=ps["poss_50ep"]["ci95_2dp"],
                per_seed=ps["poss_50ep"]["diffs"]),
         infer="marginal effect of sensor consistency at matched exposure+dose"),
    dict(label="RayMix vs. balanced-naive (sequence 03, 3x)", moves="P", protocol="seq03",
         c=paired(v3["raymix_vol3_seeds"], v3["balnaive_vol3_seeds"]),
         infer="same marginal effect at the higher dose operating point"),
    dict(label="RayMix vs. balanced-naive (KITTI, 50 ep)", moves="P", protocol="KITTI",
         c=dict(mean=ps["kitti_50ep"]["mean_2dp"], ci=ps["kitti_50ep"]["ci95_2dp"],
                per_seed=ps["kitti_50ep"]["diffs"]),
         infer="cross-sensor replication of the consistency contrast, converged"),
    dict(label="Balanced-naive vs. PolarMix (sequence 03)", moves="E", protocol="seq03",
         c=paired(arm["balanced-naive"]["seeds"], arm["PolarMix"]["seeds"]),
         infer="class-exposure balancing at matched (absent) physics"),
    dict(label="RayMix 3x vs. 1x paste volume (sequence 03)", moves="Q", protocol="seq03",
         c=paired(arm["vol_x3"]["seeds"], arm["RayMix-full"]["seeds"]),
         infer="augmentation dose at matched physics+exposure policy"),
    dict(label="RayMix vs. balanced-naive (KITTI, 24 ep)", moves="P|H", protocol="KITTI",
         c=dict(mean=ps["kitti_24ep"]["mean_2dp"], ci=ps["kitti_24ep"]["ci95_2dp"],
                per_seed=ps["kitti_24ep"]["diffs"]),
         infer="short-schedule reversal: why convergence control (H) is mandatory"),
]

# ---- cross-assertions against published numbers ----
assert rows[0]["c"]["mean"] == 2.42, rows[0]
assert rows[3]["c"]["mean"] == 0.04 and rows[3]["c"]["ci"] == [-2.33, 2.41], rows[3]
assert abs(rows[5]["c"]["mean"] - 0.05) < 0.005, rows[5]
assert rows[6]["c"]["mean"] == 0.86, rows[6]

# residual summary used in the matrix caption (P column magnitudes)
res = {}
for cond in ("naive_x3", "balanced_naive", "raymix"):
    c = sp["conditions"][cond]
    res[cond] = dict(
        vis=c["visibility"]["excess_per_10k"]["point"],
        vis_ci=[c["visibility"]["excess_per_10k"]["lo"], c["visibility"]["excess_per_10k"]["hi"]],
        behind=c["visibility"]["kept_behind_frac"]["point"],
        behind_ci=[c["visibility"]["kept_behind_frac"]["lo"],
                   c["visibility"]["kept_behind_frac"]["hi"]],
        w=c["ground"]["wasserstein_vs_real"]["point"],
        w_ci=[c["ground"]["wasserstein_vs_real"]["lo"], c["ground"]["wasserstein_vs_real"]["hi"]],
        samp=c["sampling"]["median_abs"]["point"],
        samp_ci=[c["sampling"]["median_abs"]["lo"], c["sampling"]["median_abs"]["hi"]])
floors = dict(vis=sp["conditions"]["real_floor"]["visibility"]["floor_per_10k"]["point"],
              samp=sp["conditions"]["real_floor"]["sampling"]["median_abs"])

out = dict(t_crit=T95, rows=rows, residuals_poss=res, floors_poss=floors,
           verdicts_poss=sp["verdicts"], verdicts_kitti=sk["verdicts"])
json.dump(out, open(os.path.join(PO, "scua_matrix.json"), "w"), indent=1)

# ---- LaTeX matrix body ----
def tex_row(r):
    mv = {"P": ("\\cmark", "--", "--", "--"), "E": ("--", "\\cmark", "--", "--"),
          "Q": ("--", "--", "\\cmark", "--"), "P|H": ("\\cmark", "--", "--", "\\cmark"),
          "E+Q+P": ("\\cmark", "\\cmark", "\\cmark", "--")}[r["moves"]]
    c = r["c"]
    sig = not (c["ci"][0] <= 0.0 <= c["ci"][1])
    val = f"${c['mean']:+.2f}$ $[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]$"
    if sig:
        val = f"\\textbf{{{val}}}"
    return f"{r['label']} & {mv[0]} & {mv[1]} & {mv[2]} & {mv[3]} & {val} \\\\"


tex = ["% AUTO-GENERATED by gen_scua_assets.py from machine-printed JSONs — do not hand-edit.",
       "\\begin{table*}[t]\\centering\\footnotesize",
       "\\setlength{\\tabcolsep}{3.5pt}",
       "\\caption{\\textbf{Matched identification matrix.} Each contrast changes the checked"
       " variable(s) and holds the others; $\\Delta$mIoU is the paired converged effect"
       " (best-val, Student's $t$ $95\\%$ CI, $n{=}3$ seeds; bold $=$ CI excludes zero)."
       " $P$ magnitudes for the RayMix-vs.-balanced-naive rows are the audited residual"
       " reductions of Table~\\ref{tab:scua}. Rows are machine-generated from the underlying"
       " JSONs.}",
       "\\label{tab:matrix}",
       "\\begin{tabular}{lccccl}",
       "\\toprule",
       "Contrast & $\\Delta P$ & $\\Delta E$ & $\\Delta Q$ & $\\Delta H$ &"
       " converged $\\Delta$mIoU $[95\\%$ CI$]$ \\\\ \\midrule"]
tex += [tex_row(r) for r in rows]
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(os.path.join(PO, "scua_matrix.tex"), "w").write("\n".join(tex) + "\n")

# ---- figure ----
matplotlib.rcParams.update({
    "font.size": 9, "font.family": "serif", "font.serif": ["DejaVu Serif", "Times"],
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "stix"})
C_NAIVE, C_BAL, C_RAY = "#5b8db8", "#7f8c8d", "#c0392b"
CONDS = [("naive_x3", "naive $\\times$3", C_NAIVE), ("balanced_naive", "balanced-naive", C_BAL),
         ("raymix", "RayMix", C_RAY)]

fig = plt.figure(figsize=(7.05, 3.3))
# cols: 0-1 = residual mini-panels; 2 = empty gutter reserved for the forest's row labels; 3 = forest
gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.55, 2.05], wspace=0.42, hspace=0.80,
                      left=0.115, right=0.955, top=0.885, bottom=0.145)
panels = [("vis", "vis_ci", "(a) first-return conflicts\nper 10k affected rays", floors.get("vis"), "{:.0f}"),
          ("behind", "behind_ci", "(b) background kept\nbehind insert (frac)", None, "{:.2f}"),
          ("w", "w_ci", "(c) ground-gap $W_1$\nvs. real (m)", None, "{:.2f}"),
          ("samp", "samp_ci", "(d) range-sampling\nresidual $|\\log_2|$", floors.get("samp"), "{:.2f}")]
axpos = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
         fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
for ax, (key, cikey, ttl, floor, fmt) in zip(axpos, panels):
    vals = [res[c][key] for c, _, _ in CONDS]
    cis = [res[c][cikey] for c, _, _ in CONDS]
    xmax = max(max(vals), floor or 0)
    ys = np.arange(len(CONDS))[::-1]
    for y, v, ci, (_, lbl, col) in zip(ys, vals, cis, CONDS):
        ax.barh(y, v, height=0.62, color=col, edgecolor="k", lw=0.4, zorder=3)
        ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                    ecolor="0.15", elinewidth=0.8, capsize=1.6, zorder=4)
        ax.text(v + 0.045 * xmax, y, fmt.format(v), va="center", fontsize=6.3, zorder=5)
    if floor is not None:
        ax.axvline(floor, color="0.4", ls=":", lw=0.9, zorder=2)
        ax.text(floor, -0.78, "real", fontsize=5.8, color="0.35", ha="center", va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _, l, _ in CONDS] if ax in (axpos[0], axpos[2]) else [])
    ax.set_xlim(0, xmax * 1.42)
    ax.set_ylim(-0.62, len(CONDS) - 0.38)
    ax.set_title(ttl, fontsize=6.8, loc="left", pad=3)
    ax.tick_params(axis="y", length=0)

axf = fig.add_subplot(gs[:, 3])
groups = [("COMPOSITE", [0, 1]), ("SENSOR CONSISTENCY (P)", [2, 3, 4]),
          ("EXPOSURE (E)", [5]), ("DOSE (Q)", [6]), ("CONVERGENCE (H)", [7])]
yt, yl, hdr_rows = [], [], []
y = 0.0
axf.axvspan(-0.5, 0.5, color="0.93", zorder=1)
axf.axvline(0, color="0.35", lw=0.9, zorder=2)
for gname, idxs in groups:
    yt.append(y)
    yl.append(gname)
    hdr_rows.append(len(yt) - 1)
    y += 0.9
    for i in idxs:
        r = rows[i]
        c = r["c"]
        sig = not (c["ci"][0] <= 0.0 <= c["ci"][1])
        axf.errorbar(c["mean"], y, xerr=[[c["mean"] - c["ci"][0]], [c["ci"][1] - c["mean"]]],
                     fmt="o", ms=4.0, mfc=("#c0392b" if sig else "white"), mec="#c0392b",
                     ecolor="0.2", elinewidth=1.0, capsize=1.8, zorder=4)
        axf.annotate(f"{c['mean']:+.2f}", xy=(1.015, y), xycoords=("axes fraction", "data"),
                     fontsize=6.3, va="center", ha="left",
                     fontweight="bold" if sig else "normal")
        yt.append(y)
        yl.append(r["label"])
        y += 0.9
    y += 0.30
axf.set_yticks(yt)
axf.set_yticklabels(yl, fontsize=6.3)
for i, lab in enumerate(axf.get_yticklabels()):
    if i in hdr_rows:
        lab.set_fontsize(5.6)
        lab.set_color("0.30")
        lab.set_fontweight("bold")
axf.set_ylim(y - 0.65, -0.65)
axf.set_xlim(-2.7, 3.2)
axf.set_xticks([-2, -1, 0, 1, 2, 3])
axf.set_xlabel("converged $\\Delta$mIoU (pp), paired 95% CI\n(shaded band $=$ SESOI $\\pm$0.5)",
               fontsize=7.0)
axf.tick_params(axis="y", length=0)
axf.set_title("(e) matched converged-utility contrasts", fontsize=6.8, loc="left", pad=3)

fig.savefig(os.path.join(PO, "fig_scua.pdf"))
fig.savefig(os.path.join(PO, "fig_scua.png"))
print("wrote scua_matrix.json, scua_matrix.tex, fig_scua.pdf")
for r in rows:
    print(f"  {r['moves']:6s} {r['label']:44s} {r['c']['mean']:+.2f} {r['c']['ci']}")
