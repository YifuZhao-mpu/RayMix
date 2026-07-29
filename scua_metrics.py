"""SCUA (Sensor-Consistency-Utility Audit) residual metrics — the measurement half of the
protocol-controlled study. Three quasi-independent acquisition-consistency residuals, each
sensor-calibrated and reported against a real-scan floor, per Codex-max review spec (2026-07-19):

  (1) visibility / first-return residual: per calibrated (beam, azimuth) ray cell, range conflicts
      per 10k ADDED-affected rays, with the host-alone floor subtracted; plus the fraction of host
      background points wrongly kept behind an inserted surface.
  (2) ground-support residual: SIGNED gap (base_z - local ground z) distributions per class,
      compared to real instances via 1-Wasserstein distance + floating/buried tail fractions,
      with the ground-estimator uncertainty reported.
  (3) range-sampling residual: log2 ratio of an added instance's point count to the real
      median count at the same (class, range bucket) — detects density-inconsistent placement.

  NO weighted total score (the three residuals have different engineering consequences).
  All summaries carry scan-level bootstrap 95% CIs. Machine-printed JSON only.

Beam model: the uniform-elevation binning used inside RayMix is NOT assumed; beams are calibrated
empirically per sensor by 1-D k-means on elevation angles over many real scans, and the calibration
quality (intra-beam spread vs inter-beam gap) is itself reported.
"""
import numpy as np


# ---------------------------------------------------------------------------
# sensor calibration
# ---------------------------------------------------------------------------
def calibrate_beams(elev_samples, k, lo_deg, hi_deg, iters=40, max_pts=2_000_000, seed=0):
    """1-D k-means (quantile init, order-preserving) on elevation angles pooled over real scans.
    Returns dict with sorted beam centers (rad), bin edges, and separation diagnostics."""
    rng = np.random.default_rng(seed)
    x = np.asarray(elev_samples, np.float64)
    lo, hi = np.radians(lo_deg), np.radians(hi_deg)
    x = x[(x > lo) & (x < hi)]
    if len(x) > max_pts:
        x = rng.choice(x, max_pts, replace=False)
    centers = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        edges = (centers[1:] + centers[:-1]) / 2
        idx = np.searchsorted(edges, x)
        sums = np.bincount(idx, weights=x, minlength=k)
        cnts = np.bincount(idx, minlength=k)
        upd = cnts > 0
        centers[upd] = sums[upd] / cnts[upd]
        centers = np.sort(centers)
    edges = (centers[1:] + centers[:-1]) / 2
    idx = np.searchsorted(edges, x)
    intra = np.array([x[idx == j].std() if (idx == j).sum() > 1 else 0.0 for j in range(k)])
    gaps = np.diff(centers)
    sep = float(np.median(gaps) / max(np.median(intra[intra > 0]), 1e-9))
    return dict(centers=centers, edges=edges, k=k,
                intra_beam_std_med_deg=float(np.degrees(np.median(intra[intra > 0]))),
                inter_beam_gap_med_deg=float(np.degrees(np.median(gaps))),
                separation_ratio=sep,
                occupancy_min=int(np.bincount(idx, minlength=k).min()))


class RayModel:
    """Calibrated (beam, azimuth) cell assignment."""

    def __init__(self, beam_edges, n_beam, n_azimuth):
        self.edges = np.asarray(beam_edges, np.float64)
        self.n_beam = n_beam
        self.n_az = n_azimuth

    def cells(self, coord):
        rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
        r = np.sqrt(rho ** 2 + coord[:, 2] ** 2) + 1e-6
        theta = np.arctan2(coord[:, 1], coord[:, 0])
        phi = np.arctan2(coord[:, 2], rho + 1e-6)
        beam = np.searchsorted(self.edges, phi)
        az = np.clip(((theta + np.pi) / (2 * np.pi) * self.n_az).astype(np.int64), 0, self.n_az - 1)
        return beam * self.n_az + az, r.astype(np.float32)

    @property
    def n_cells(self):
        return self.n_beam * self.n_az


# ---------------------------------------------------------------------------
# residual (1): visibility / first-return
# ---------------------------------------------------------------------------
def visibility_residual(ray, host_c, add_c, host_final_c=None, add_final_c=None, tol=0.5):
    """Per scan. If add_c is None: real-scan floor = conflict cells per 10k occupied cells.
    Else: conflicts per 10k insertion-affected rays in the FINAL scan the method produces
    (host_final + add_final), minus the original-host rate over the same rays; plus the fraction
    of surviving background points wrongly kept behind a surviving inserted surface."""
    hc, hr = ray.cells(host_c)
    if add_c is None or len(add_c) == 0:
        occ = np.unique(hc)
        conf = _conflict_cells(hc, hr, ray.n_cells, tol)
        return dict(kind="floor", affected=int(len(occ)),
                    conflicts_per_10k=1e4 * len(conf) / max(len(occ), 1))
    if host_final_c is None:
        host_final_c = host_c
    if add_final_c is None:
        add_final_c = add_c
    if len(add_final_c) == 0:
        return None
    ac, ar = ray.cells(add_final_c)
    affected = np.unique(ac)
    hfc, hfr = ray.cells(host_final_c)
    fc = np.concatenate([hfc, ac]); fr = np.concatenate([hfr, ar])
    conf_final = _conflict_cells(fc, fr, ray.n_cells, tol, restrict=affected)
    conf_host = _conflict_cells(hc, hr, ray.n_cells, tol, restrict=affected)
    # background wrongly kept behind a surviving inserted surface, as a fraction of the
    # background ORIGINALLY on the inserted rays (stable denominator across merge policies)
    amin = np.full(ray.n_cells, np.inf, np.float32)
    np.minimum.at(amin, ac, ar)
    n_host_aff = int(np.isin(hc, affected).sum())
    in_aff_fin = np.isin(hfc, affected)
    behind = int((hfr[in_aff_fin] > amin[hfc[in_aff_fin]] + tol).sum())
    return dict(kind="added",
                affected=int(len(affected)),
                conflicts_per_10k=1e4 * len(conf_final) / max(len(affected), 1),
                host_floor_per_10k=1e4 * len(conf_host) / max(len(affected), 1),
                excess_per_10k=1e4 * (len(conf_final) - len(conf_host)) / max(len(affected), 1),
                kept_behind_frac=behind / max(n_host_aff, 1))


def _conflict_cells(cells, r, n_cells, tol, restrict=None):
    rmin = np.full(n_cells, np.inf, np.float32)
    rmax = np.full(n_cells, -np.inf, np.float32)
    np.minimum.at(rmin, cells, r)
    np.maximum.at(rmax, cells, r)
    conf = np.nonzero((rmax - rmin) > tol)[0]
    if restrict is not None:
        conf = conf[np.isin(conf, restrict)]
    return conf


# ---------------------------------------------------------------------------
# residual (2): ground support (signed)
# ---------------------------------------------------------------------------
def signed_ground_gap(host_c, ground_mask, inst_c, radius=2.0, min_ground=5):
    """Signed gap for ONE instance: base_z - median local ground z (None if estimator undefined).
    `ground_mask` is a boolean mask over host points (supports multi-class ground, e.g. KITTI).
    Also returns the estimator spread (std of local ground z)."""
    if not ground_mask.any():
        return None
    gz = host_c[ground_mask]
    cen = inst_c[:, :2].mean(0)
    near = gz[np.sum((gz[:, :2] - cen) ** 2, 1) < radius ** 2]
    if len(near) < min_ground:
        return None
    return dict(gap=float(inst_c[:, 2].min() - np.median(near[:, 2])),
                est_std=float(near[:, 2].std()))


def ground_summary(gaps, real_gaps, float_tol=0.30, bury_tol=-0.15):
    """Distribution summary + 1-Wasserstein distance to the real signed-gap distribution."""
    g = np.asarray(gaps, np.float64)
    if len(g) == 0:
        return None
    out = dict(n=int(len(g)), median=float(np.median(g)),
               iqr=float(np.percentile(g, 75) - np.percentile(g, 25)),
               float_frac=float((g > float_tol).mean()),
               bury_frac=float((g < bury_tol).mean()))
    if real_gaps is not None and len(real_gaps) > 3:
        out["wasserstein_vs_real"] = float(_w1(g, np.asarray(real_gaps, np.float64)))
    return out


def _w1(a, b, n=512):
    q = np.linspace(0.01, 0.99, n)
    return np.abs(np.quantile(a, q) - np.quantile(b, q)).mean()


# ---------------------------------------------------------------------------
# residual (3): range-sampling
# ---------------------------------------------------------------------------
class RangeSamplingModel:
    """Real-data reference: per (class, range bucket) median log2 point count."""

    def __init__(self, bucket_m=10.0, max_bucket=50.0):
        self.bucket_m = bucket_m
        self.max_bucket = max_bucket
        self.obs = {}

    def bucket(self, r):
        return min(int(r // self.bucket_m) * int(self.bucket_m), int(self.max_bucket))

    def add_real(self, cls, rng, npts):
        self.obs.setdefault((int(cls), self.bucket(rng)), []).append(np.log2(npts))

    def finalize(self, min_n=5):
        self.med = {k: float(np.median(v)) for k, v in self.obs.items() if len(v) >= min_n}
        self.counts = {f"{k[0]}_{k[1]}": len(v) for k, v in self.obs.items()}

    def residual(self, cls, rng, npts):
        m = self.med.get((int(cls), self.bucket(rng)))
        return None if m is None else float(np.log2(npts) - m)


def sampling_summary(residuals):
    x = np.asarray([r for r in residuals if r is not None], np.float64)
    if len(x) == 0:
        return None
    return dict(n=int(len(x)), median_abs=float(np.median(np.abs(x))),
                bias=float(np.mean(x)), frac_gt_1=float((np.abs(x) > 1.0).mean()))


# ---------------------------------------------------------------------------
# scan-level bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(per_scan_values, stat=np.mean, B=2000, seed=0):
    """95% CI by resampling SCANS (not points/instances) with replacement."""
    v = [x for x in per_scan_values if x is not None]
    if len(v) < 3:
        return None
    v = np.asarray(v, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v)))
    boots = stat(v[idx], axis=1)
    return dict(point=float(stat(v)), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n_scans=int(len(v)))
