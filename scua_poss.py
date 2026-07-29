"""SCUA controlled-corruption calibration harness — SemanticPOSS (Pandora 40-beam).

Conditions, all sharing the same host/donor pairing and donor-instance extraction per scan:
  real_floor      host scan alone (visibility floor; also the source pass for the real ground-gap
                  distribution and the real point-count-vs-range reference model)
  polarmix_visonly  PolarMix's true instance branch (WHOLE class point sets x3 fixed yaws, no
                    physics); scan-level visibility metrics only
  naive_x3        unbalanced-volume naive control: each bank instance x3 fixed yaws, no physics
  balanced_naive  RayMix bank/round-robin sampling; snap+validity+z-buffer OFF (paper's control arm)
  raymix          full pipeline (snap, validity, ray z-buffer)
  raymix_lift{03,06,12}   full placement, then +0.3/0.6/1.2 m lift BEFORE z-buffer
                          -> ground residual must respond monotonically; others must not
  raymix_nozbuf   full placement (snap+validity) but naive concat (z-buffer off)
                          -> visibility residual must respond; others must not
  raymix_shift{10,20}     full placement, then radial +10/+20 m xy-shift, z re-adjusted to keep the
                          instance's signed ground gap at the new location (no density adaptation)
                          -> range-sampling residual must respond; others must not

Machine-prints outputs/poss/scua_poss.json (calibration, per-condition residuals with scan-level
bootstrap CIs, and machine-evaluated monotonicity/specificity verdicts). No training, CPU only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/project/Pointcept")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import DBSCAN  # noqa: E402

from pointcept.datasets.semantic_poss_raymix import (  # noqa: E402
    GROUND_SUPPORTED, SemanticPOSSRayMixDataset, _rotate_z)
import scua_metrics as M  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "poss")
ROOT = "/root/project/data/SemanticPOSS"
N_SCANS = int(os.environ.get("SCUA_N", 120))
N_CALIB = int(os.environ.get("SCUA_CALIB_N", 60))
N_AZ = 1800
GROUND_CLASS = 12
CONDS = ["polarmix_visonly", "naive_x3", "balanced_naive", "raymix", "raymix_lift03",
         "raymix_lift06", "raymix_lift12", "raymix_nozbuf", "raymix_shift10", "raymix_shift20"]
LIFT = {"raymix_lift03": 0.3, "raymix_lift06": 0.6, "raymix_lift12": 1.2}
SHIFT = {"raymix_shift10": 10.0, "raymix_shift20": 20.0}


def elev(coord):
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    return np.arctan2(coord[:, 2], rho + 1e-6)


def real_instances(ds, c, t):
    """DBSCAN real instances of the transplant classes in a scan -> (cls, mean range, npts, coords)."""
    out = []
    for cls in ds.transplant_classes:
        m = t == cls
        if m.sum() < ds.min_inst_pts:
            continue
        pts = c[m]
        lab = DBSCAN(eps=ds.dbscan_eps, min_samples=ds.dbscan_min).fit(pts).labels_
        for k in np.unique(lab):
            if k == -1:
                continue
            ip = pts[lab == k]
            if len(ip) < ds.min_inst_pts:
                continue
            rng = float(np.sqrt((ip[:, :2] ** 2).sum(1)).mean())
            out.append((int(cls), rng, len(ip), ip))
    return out


def place_condition(ds, cond, insts, cH, tH, rng, donor=None, gmask=None):
    """Produce (add_c, add_t, per-instance records, merge_mode) for one condition."""
    if gmask is None:
        gmask = tH == GROUND_CLASS
    recs, add_c, add_t = [], [], []
    if cond == "polarmix_visonly":
        # true PolarMix instance branch: WHOLE class point sets x3 fixed yaws, no physics.
        # Scan-level visibility metrics only (no DBSCAN instances -> no gap/sampling records).
        cD, tD = donor
        m = np.isin(tD, ds.transplant_classes)
        if not m.any():
            return [], [], [], "concat"
        ci, ti = cD[m], tD[m].astype(np.int32)
        for th in (np.pi / 2, np.pi, np.pi * 3 / 2):
            add_c.append(_rotate_z(ci, th).astype(np.float32))
            add_t.append(ti)
        return add_c, add_t, [], "concat"
    if cond == "naive_x3":
        # unbalanced-volume naive control: each bank instance at 3 fixed yaws, no physics
        for (ip, ist, cls, base_z, clear) in insts:
            for th in (np.pi / 2, np.pi, np.pi * 3 / 2):
                rp = _rotate_z(ip, th).astype(np.float32)
                add_c.append(rp)
                add_t.append(np.full(len(rp), cls, np.int32))
                recs.append(dict(cls=cls, pts=rp))
        return add_c, add_t, recs, "concat"
    snap = cond != "balanced_naive"
    ds.snap_ground = snap
    ds.validity = snap
    for inst in insts:
        placed = ds._place(inst, cH, tH)
        if placed is None:
            continue
        pc, ps, cls = placed
        if cond in LIFT:
            pc = pc.copy()
            pc[:, 2] += LIFT[cond]
        if cond in SHIFT:
            pc = radial_shift(pc, SHIFT[cond], cH, gmask)
            if pc is None:
                continue
        add_c.append(pc)
        add_t.append(np.full(len(pc), cls, np.int32))
        recs.append(dict(cls=cls, pts=pc))
    merge = "zbuf" if cond in ("raymix",) or cond in LIFT or cond in SHIFT else "concat"
    return add_c, add_t, recs, merge


def radial_shift(pc, dist, cH, gmask, radius=2.0):
    """Shift an instance radially outward by `dist` m; re-adjust z to preserve its signed ground gap
    at the NEW location (so only the range-sampling residual should respond). None if no ground."""
    cen = pc[:, :2].mean(0)
    r0 = np.linalg.norm(cen)
    if r0 < 1e-3:
        return None
    g0 = M.signed_ground_gap(cH, gmask, pc, radius)
    out = pc.copy()
    out[:, :2] += (cen / r0) * dist
    if g0 is not None:
        g1 = M.signed_ground_gap(cH, gmask, out, radius)
        if g1 is None:
            return None
        out[:, 2] += g0["gap"] - g1["gap"]
    return out


def pooled_bootstrap(scan_lists, stat, B=2000, seed=0):
    """Bootstrap a pooled statistic by resampling SCANS with replacement."""
    scan_lists = [s for s in scan_lists if len(s)]
    if len(scan_lists) < 3:
        return None
    rng = np.random.default_rng(seed)
    point = stat(np.concatenate(scan_lists))
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(scan_lists), len(scan_lists))
        boots.append(stat(np.concatenate([scan_lists[i] for i in idx])))
    return dict(point=float(point), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n_scans=len(scan_lists))


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SemanticPOSSRayMixDataset(split="train", data_root=ROOT, eval_seq=3,
                                   raymix_p=1.0, swap_p=0.0, transform=[])
    rng = np.random.default_rng(11)

    # ---- step 0: sensor calibration (empirical beam model) ----
    calib_idx = rng.choice(len(ds.data_list), N_CALIB, replace=False)
    elevs = [elev(ds._load_raw(ds.data_list[i])[0]) for i in calib_idx]
    cal = M.calibrate_beams(np.concatenate(elevs), k=40, lo_deg=-16.5, hi_deg=8.5)
    ray = M.RayModel(cal["edges"], n_beam=40, n_azimuth=N_AZ)
    print(f"[calib] beams=40 sep_ratio={cal['separation_ratio']:.1f} "
          f"intra={cal['intra_beam_std_med_deg']:.4f}deg gap={cal['inter_beam_gap_med_deg']:.4f}deg")

    # ---- step 1: real reference pass (floor, real gaps, sampling model) ----
    host_idx = rng.choice(len(ds.data_list), N_SCANS, replace=False)
    donor_idx = np.roll(host_idx, 37)
    smodel = M.RangeSamplingModel()
    floor_scan, real_gap_scan, real_triplets = [], [], []
    cache = {}
    for n, (hi, di) in enumerate(zip(host_idx, donor_idx)):
        cH, sH, tH = ds._load_raw(ds.data_list[hi])
        cD, sD, tD = ds._load_raw(ds.data_list[di])
        floor_scan.append(M.visibility_residual(ray, cH, None)["conflicts_per_10k"])
        gaps = []
        for (cls, rg, npts, ip) in real_instances(ds, cH, tH):
            smodel.add_real(cls, rg, npts)
            real_triplets.append((cls, rg, npts))
            if cls in GROUND_SUPPORTED:
                g = M.signed_ground_gap(cH, tH == GROUND_CLASS, ip, ds.ground_radius)
                if g is not None:
                    gaps.append(g["gap"])
        real_gap_scan.append(np.asarray(gaps))
        cache[hi] = (cH, sH, tH, ds._extract_instances(cD, sD, tD), (cD, tD))
        if (n + 1) % 20 == 0:
            print(f"[real] {n+1}/{N_SCANS}")
    smodel.finalize()
    real_gaps_all = np.concatenate(real_gap_scan)
    # real-instance sampling self-residual = the natural floor of the range-sampling metric
    real_sres = [smodel.residual(c, r, n) for (c, r, n) in real_triplets]
    real_sres = np.asarray([x for x in real_sres if x is not None])

    # ---- step 2: conditions ----
    results = {"real_floor": dict(
        visibility=dict(floor_per_10k=M.bootstrap_ci(floor_scan)),
        ground=M.ground_summary(real_gaps_all, None),
        sampling=M.sampling_summary(real_sres))}
    for cond in CONDS:
        vis_ex, vis_bh, gap_sc, samp_sc = [], [], [], []
        np.random.seed(20260719)  # placement RNG reproducible & shared across conditions
        for hi in host_idx:
            cH, sH, tH, insts, donor = cache[hi]
            add_c, add_t, recs, merge = place_condition(ds, cond, insts, cH, tH, rng, donor)
            if not add_c:
                continue
            A_c = np.concatenate(add_c, 0)
            A_t = np.concatenate(add_t, 0)
            if merge == "zbuf":
                # replicate _ray_merge's split so host/added parts of the final scan stay separable
                # (the method's own uniform-bin ray model decides survival; measurement stays calibrated)
                cellA, rA = ds._cell(cH)
                cellI, rI = ds._cell(A_c)
                mx = ds.n_beam * ds.n_azimuth
                host_min = np.full(mx, np.inf, np.float32)
                np.minimum.at(host_min, cellA, rA)
                inst_min = np.full(mx, np.inf, np.float32)
                np.minimum.at(inst_min, cellI, rI)
                host_fin = cH[~(rA > inst_min[cellA])]
                add_fin = A_c[rI < host_min[cellI]]
            else:
                host_fin, add_fin = cH, A_c
            v = M.visibility_residual(ray, cH, A_c, host_final_c=host_fin,
                                      add_final_c=add_fin, tol=0.5)
            if v is None:
                continue
            vis_ex.append(v["excess_per_10k"])
            vis_bh.append(v["kept_behind_frac"])
            gaps, samps = [], []
            for rec in recs:
                # gap + sampling are PLACEMENT-quality constructs: measured as-placed, before the
                # z-buffer, so occlusion effects stay attributed to the visibility residual only.
                pc, cls = rec["pts"], rec["cls"]
                if cls in GROUND_SUPPORTED:
                    g = M.signed_ground_gap(cH, tH == GROUND_CLASS, pc, ds.ground_radius)
                    if g is not None:
                        gaps.append(g["gap"])
                rg = float(np.sqrt((pc[:, :2] ** 2).sum(1)).mean())
                res = smodel.residual(cls, rg, len(pc))
                if res is not None:
                    samps.append(res)
            gap_sc.append(np.asarray(gaps))
            samp_sc.append(np.asarray(samps))
        results[cond] = dict(
            visibility=dict(excess_per_10k=M.bootstrap_ci(vis_ex),
                            kept_behind_frac=M.bootstrap_ci(vis_bh)),
            ground=dict(median=pooled_bootstrap(gap_sc, np.median),
                        float_frac=pooled_bootstrap(gap_sc, lambda x: (x > 0.30).mean()),
                        wasserstein_vs_real=pooled_bootstrap(
                            gap_sc, lambda x: M._w1(x, real_gaps_all))),
            sampling=dict(median_abs=pooled_bootstrap(samp_sc, lambda x: np.median(np.abs(x))),
                          frac_gt_1=pooled_bootstrap(samp_sc, lambda x: (np.abs(x) > 1.0).mean())))
        g = results[cond]["ground"]["median"]
        print(f"[{cond}] vis_excess={results[cond]['visibility']['excess_per_10k']['point']:.1f} "
              f"gap_med={g['point']:.3f}" if g else f"[{cond}] done")

    # ---- step 3: machine-evaluated calibration verdicts ----
    def pt(cond, grp, key):
        d = results[cond][grp][key]
        return d["point"] if d else None

    lifts = [pt(c, "ground", "median") for c in ("raymix", "raymix_lift03",
                                                 "raymix_lift06", "raymix_lift12")]
    shifts = [pt(c, "sampling", "median_abs") for c in ("raymix", "raymix_shift10",
                                                        "raymix_shift20")]

    def mono(vals):
        return None if any(v is None for v in vals) else bool(
            all(a < b for a, b in zip(vals, vals[1:])))

    verdicts = dict(
        ground_monotone_under_lift=mono(lifts),
        lift_ladder=lifts,
        visibility_responds_to_nozbuf=bool(
            pt("raymix_nozbuf", "visibility", "excess_per_10k")
            > 5 * max(pt("raymix", "visibility", "excess_per_10k"), 1e-6)),
        sampling_monotone_under_shift=mono(shifts),
        shift_ladder=shifts,
        specificity_lift_does_not_move_vis=bool(
            abs(pt("raymix_lift12", "visibility", "excess_per_10k")
                - pt("raymix", "visibility", "excess_per_10k"))
            < 0.2 * max(pt("raymix_nozbuf", "visibility", "excess_per_10k"), 1e-6)),
        specificity_shift_does_not_move_ground=bool(
            abs((pt("raymix_shift20", "ground", "median") or 0)
                - (pt("raymix", "ground", "median") or 0)) < 0.10),
        specificity_nozbuf_does_not_move_sampling=bool(
            abs(pt("raymix_nozbuf", "sampling", "median_abs")
                - pt("raymix", "sampling", "median_abs")) < 0.15))

    out = dict(sensor="Hesai-Pandora-40 (SemanticPOSS)", n_scans=N_SCANS,
               calibration={k: v for k, v in cal.items() if k not in ("centers", "edges")},
               beam_centers_deg=[round(float(np.degrees(c)), 3) for c in cal["centers"]],
               real_sampling_model_cells=smodel.counts,
               conditions=results, verdicts=verdicts)
    path = os.path.join(OUT, "scua_poss.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("VERDICTS:", json.dumps(verdicts, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
