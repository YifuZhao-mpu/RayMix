"""SCUA controlled-corruption calibration — SemanticKITTI (Velodyne HDL-64E) cross-sensor arm.

Same conditions and verdicts as scua_poss.py, with the KITTI RayMix machinery (multi-class ground,
64 calibrated beams). The point of this arm: the audit protocol transfers across sensors — POSS has
exactly-discrete beam angles, KITTI has noisy continuous ones, and the calibration diagnostics
(separation ratio) quantify beam-model confidence in each regime.
Machine-prints outputs/poss/scua_kitti.json. No training, CPU only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/project/Pointcept")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointcept.datasets.semantic_kitti_raymix import (  # noqa: E402
    K_GROUND_SUPPORTED, SemanticKITTIRayMixDataset)
import scua_metrics as M  # noqa: E402
import scua_poss as P  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "poss")
ROOT = "/datasets/data/semiantic_kitti"
N_SCANS = int(os.environ.get("SCUA_N", 60))
N_CALIB = int(os.environ.get("SCUA_CALIB_N", 40))
N_AZ = 2048


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = SemanticKITTIRayMixDataset(split="train", data_root=ROOT,
                                    raymix_p=1.0, swap_p=0.0, transform=[])
    rng = np.random.default_rng(11)

    calib_idx = rng.choice(len(ds.data_list), N_CALIB, replace=False)
    elevs = [P.elev(ds._load_raw(ds.data_list[i])[0]) for i in calib_idx]
    cal = M.calibrate_beams(np.concatenate(elevs), k=64, lo_deg=-26.0, hi_deg=4.0)
    ray = M.RayModel(cal["edges"], n_beam=64, n_azimuth=N_AZ)
    print(f"[calib] beams=64 sep_ratio={cal['separation_ratio']:.1f} "
          f"intra={cal['intra_beam_std_med_deg']:.4f}deg gap={cal['inter_beam_gap_med_deg']:.4f}deg")

    host_idx = rng.choice(len(ds.data_list), N_SCANS, replace=False)
    donor_idx = np.roll(host_idx, 37)
    smodel = M.RangeSamplingModel()
    floor_scan, real_gap_scan, real_triplets = [], [], []
    cache = {}
    for n, (hi, di) in enumerate(zip(host_idx, donor_idx)):
        cH, sH, tH = ds._load_raw(ds.data_list[hi])
        cD, sD, tD = ds._load_raw(ds.data_list[di])
        floor_scan.append(M.visibility_residual(ray, cH, None)["conflicts_per_10k"])
        gm = ds._is_ground(tH)
        gaps = []
        for (cls, rg, npts, ip) in P.real_instances(ds, cH, tH):
            smodel.add_real(cls, rg, npts)
            real_triplets.append((cls, rg, npts))
            if cls in K_GROUND_SUPPORTED:
                g = M.signed_ground_gap(cH, gm, ip, ds.ground_radius)
                if g is not None:
                    gaps.append(g["gap"])
        real_gap_scan.append(np.asarray(gaps))
        cache[hi] = (cH, sH, tH, ds._extract_instances(cD, sD, tD), (cD, tD))
        if (n + 1) % 10 == 0:
            print(f"[real] {n+1}/{N_SCANS}")
    smodel.finalize()
    real_gaps_all = np.concatenate(real_gap_scan)
    real_sres = [smodel.residual(c, r, n) for (c, r, n) in real_triplets]
    real_sres = np.asarray([x for x in real_sres if x is not None])

    results = {"real_floor": dict(
        visibility=dict(floor_per_10k=M.bootstrap_ci(floor_scan)),
        ground=M.ground_summary(real_gaps_all, None),
        sampling=M.sampling_summary(real_sres))}
    for cond in P.CONDS:
        vis_ex, vis_bh, gap_sc, samp_sc = [], [], [], []
        np.random.seed(20260719)
        for hi in host_idx:
            cH, sH, tH, insts, donor = cache[hi]
            gm = ds._is_ground(tH)
            add_c, add_t, recs, merge = P.place_condition(ds, cond, insts, cH, tH, rng,
                                                          donor, gmask=gm)
            if not add_c:
                continue
            A_c = np.concatenate(add_c, 0)
            if merge == "zbuf":
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
                pc, cls = rec["pts"], rec["cls"]
                if cls in K_GROUND_SUPPORTED:
                    g = M.signed_ground_gap(cH, gm, pc, ds.ground_radius)
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
            ground=dict(median=P.pooled_bootstrap(gap_sc, np.median),
                        float_frac=P.pooled_bootstrap(gap_sc, lambda x: (x > 0.30).mean()),
                        wasserstein_vs_real=P.pooled_bootstrap(
                            gap_sc, lambda x: M._w1(x, real_gaps_all))),
            sampling=dict(median_abs=P.pooled_bootstrap(samp_sc, lambda x: np.median(np.abs(x))),
                          frac_gt_1=P.pooled_bootstrap(samp_sc, lambda x: (np.abs(x) > 1.0).mean())))
        g = results[cond]["ground"]["median"]
        v = results[cond]["visibility"]["excess_per_10k"]
        print(f"[{cond}] vis_excess={v['point']:.1f} " +
              (f"gap_med={g['point']:.3f}" if g else "(vis-only)"))

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

    out = dict(sensor="Velodyne-HDL64E (SemanticKITTI)", n_scans=N_SCANS,
               calibration={k: v for k, v in cal.items() if k not in ("centers", "edges")},
               beam_centers_deg=[round(float(np.degrees(c)), 3) for c in cal["centers"]],
               real_sampling_model_cells=smodel.counts,
               conditions=results, verdicts=verdicts)
    path = os.path.join(OUT, "scua_kitti.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("VERDICTS:", json.dumps(verdicts, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
