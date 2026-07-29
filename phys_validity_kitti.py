"""Cross-sensor physical diagnostics on SemanticKITTI (64-beam) — Codex cross-sensor must-have, NO training.
Shows the PolarMix double-return/floating artifact and the RayMix fix transfer from POSS (40-beam) to KITTI (64-beam),
i.e. they are sensor-agnostic. Uses the SemanticKITTIRayMixDataset's own placement/occlusion methods in two modes."""
import numpy as np
from pointcept.datasets.semantic_kitti_raymix import (
    SemanticKITTIRayMixDataset, K_GROUND_SUPPORTED, K_GROUND_CLASSES)

DR = "/datasets/data/semiantic_kitti"
NB, NA = 64, 2048
ELO, EHI = np.radians(-25.0), np.radians(3.0)


def cellrange(coord):
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    r = np.sqrt(rho ** 2 + coord[:, 2] ** 2) + 1e-6
    theta = np.arctan2(coord[:, 1], coord[:, 0]); phi = np.arctan2(coord[:, 2], rho + 1e-6)
    beam = np.clip(((phi - ELO) / (EHI - ELO) * NB).astype(np.int64), 0, NB - 1)
    az = np.clip(((theta + np.pi) / (2 * np.pi) * NA).astype(np.int64), 0, NA - 1)
    return beam * NA + az, r


def double_returns(scan_c, tol=0.5):
    if len(scan_c) == 0:
        return 0
    c, r = cellrange(scan_c); mx = NB * NA
    rmin = np.full(mx, np.inf, np.float32); np.minimum.at(rmin, c, r)
    rmax = np.full(mx, -np.inf, np.float32); np.maximum.at(rmax, c, r)
    return int(((rmax - rmin) > tol).sum())


def ground_float(host_c, host_t, add_c, add_t, ds, radius=2.0):
    gm = np.isin(host_t, K_GROUND_CLASSES)
    if not gm.any():
        return []
    gz = host_c[gm]; devs = []
    for c in np.unique(add_t):
        if int(c) not in K_GROUND_SUPPORTED:
            continue
        pts = add_c[add_t == c]
        if len(pts) < 5:
            continue
        cen = pts[:, :2].mean(0); base = pts[:, 2].min()
        near = gz[np.sum((gz[:, :2] - cen) ** 2, 1) < radius ** 2]
        if len(near):
            devs.append(abs(float(base) - float(np.median(near[:, 2]))))
    return devs


def main(N=30):
    ds = SemanticKITTIRayMixDataset(split="train", data_root=DR, raymix_p=1.0, swap_p=1.0, transform=[])
    np.random.seed(5); idxs = np.random.choice(len(ds.data_list), N, replace=False)
    pm_dr, rm_dr, pm_gf, rm_gf, real_gf = [], [], [], [], []
    for j in idxs:
        cH, sH, tH = ds._load_raw(ds.data_list[j])
        cD, sD, tD = ds._load_raw(ds.data_list[np.random.randint(len(ds.data_list))])
        host_base = double_returns(cH)
        insts = ds._extract_instances(cD, sD, tD)
        # naive: rotate+concat (no snap/validity/merge)
        nc, nt = [], []
        for it in insts:
            ip, ist, c, *_ = it
            th = np.random.uniform(-np.pi, np.pi)
            co, sn = np.cos(th), np.sin(th)
            R = np.array([[co, -sn, 0], [sn, co, 0], [0, 0, 1]], np.float32)
            nc.append(ip @ R.T); nt.append(np.full(len(ip), c, np.int32))
        if nc:
            A = np.concatenate(nc, 0); At = np.concatenate(nt, 0)
            pm_dr.append(double_returns(np.concatenate([cH, A], 0)) - host_base)
            pm_gf += ground_float(cH, tH, A, At, ds)
        # RayMix: place (snap+validity) + ray z-buffer
        a_c, a_t = [], []
        for it in insts:
            pl = ds._place(it, cH, tH)
            if pl is None:
                continue
            a_c.append(pl[0]); a_t.append(np.full(len(pl[0]), pl[2], np.int32))
        if a_c:
            A = np.concatenate(a_c, 0); At = np.concatenate(a_t, 0)
            merged = ds._ray_merge(cH, sH, tH, A, np.zeros((len(A), 1), np.float32), At)
            rm_dr.append(double_returns(merged[0]) - host_base)
            rm_gf += ground_float(cH, tH, A, At, ds)
        # real reference
        from sklearn.cluster import DBSCAN
        for c in K_GROUND_SUPPORTED:
            m = tH == c
            if m.sum() < 20:
                continue
            lab = DBSCAN(eps=0.5, min_samples=10).fit(cH[m]).labels_; P = cH[m]
            for k in np.unique(lab):
                if k == -1:
                    continue
                pts = P[lab == k]
                if len(pts) < 20:
                    continue
                cen = pts[:, :2].mean(0); base = pts[:, 2].min()
                gz = cH[np.isin(tH, K_GROUND_CLASSES)]
                near = gz[np.sum((gz[:, :2] - cen) ** 2, 1) < 4.0]
                if len(near):
                    real_gf.append(abs(float(base) - float(np.median(near[:, 2]))))
    st = lambda x: f"med {np.median(x):.3f} mean {np.mean(x):.3f} n={len(x)}" if x else "n=0"
    print("\n========= CROSS-SENSOR PHYSICAL DIAGNOSTICS: SemanticKITTI 64-beam (%d scans) =========" % N)
    print(f"(A) ray double-returns ADDED/scan:  PolarMix-naive {np.mean(pm_dr):.0f}   RayMix {np.mean(rm_dr):.0f}  (->~0)")
    print(f"(B) ground-clearance (m): REAL {st(real_gf)} | PolarMix-naive {st(pm_gf)} | RayMix {st(rm_gf)}")
    print("=> if PolarMix>>RayMix on (A) and RayMix≈REAL<<PolarMix on (B), the artifact/fix is sensor-agnostic.")
    print("================================================================================\n")
    import json
    json.dump(dict(double_returns=dict(polarmix=float(np.mean(pm_dr)), raymix=float(np.mean(rm_dr))),
                   ground_float=dict(real=float(np.median(real_gf)) if real_gf else None,
                                     polarmix=float(np.median(pm_gf)) if pm_gf else None,
                                     raymix=float(np.median(rm_gf)) if rm_gf else None)),
              open("/root/Fresh_ARIS8/code/outputs/poss/phys_validity_kitti.json", "w"), indent=2)


if __name__ == "__main__":
    main()
