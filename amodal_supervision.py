"""Amodal multi-sweep supervision for single-scan LiDAR seg (the novel core).
For each VISIBLE point p of the current scan (ray = p/|p| from sensor origin), the region BEYOND p along the
ray is OCCLUDED in this scan. Across ±K pose-registered sweeps, some of that hidden region was observed from
other viewpoints. We build a per-(visible-point) AMODAL TARGET: (a) is there hidden-occupied space behind p?
(b) the semantic class of the nearest hidden surface behind p. A train-time aux head predicts these, forcing
the backbone to encode amodal context -> better visible-point features (esp. far/occluded/sparse THING points).

Target construction (per visible point p):
  - ray direction u = p/|p|; r_p = |p|.
  - candidate hidden points = accumulated points q with: |q|-projection along u  > r_p + MARGIN  (behind p),
    and angular offset between q and the ray < ANG_TOL (same ray cone). Exclude points of the current scan.
  - hidden_occ = 1 if >=1 candidate within HID_RANGE behind p, else 0.
  - hidden_cls = majority semantic class of the nearest few candidates (only where hidden_occ=1; else IGNORE).
Static-scene assumption (accumulation valid for static structure); dynamic objects add noise -> reported.
"""
import numpy as np
from scipy.spatial import cKDTree

IGNORE = -1
ANG_TOL = 0.015      # rad (~0.86 deg) ray cone half-angle
MARGIN = 0.3         # m, must be at least this far behind p to count as "hidden"
HID_RANGE = 8.0      # m, look at most this far behind p
KNN = 5              # nearest hidden candidates for class vote


def build_amodal_targets(coord, seg, acc_coord, acc_seg):
    """coord (N,3) current visible points; seg (N,) labels 0..C-1/-1; acc_coord (M,3) accumulated points in the
    SAME frame; acc_seg (M,) their labels. Returns hidden_occ (N,) {0,1}, hidden_cls (N,) {0..C-1 / IGNORE}."""
    N = coord.shape[0]
    hidden_occ = np.zeros(N, np.int64)
    hidden_cls = np.full(N, IGNORE, np.int64)
    r = np.linalg.norm(coord, axis=1) + 1e-6
    u = coord / r[:, None]                                   # (N,3) ray dirs
    ra = np.linalg.norm(acc_coord, axis=1) + 1e-6
    # project accumulated points onto each ray is O(N*M) — too big. Use a KD-tree on direction-normalized
    # accumulated points: points on the same ray have nearly-identical unit direction. Query nearest dirs.
    ua = acc_coord / ra[:, None]                            # (M,3) unit dirs
    tree = cKDTree(ua)
    # for each visible point, find accumulated points whose unit-dir is within chord(ANG_TOL) of u_p
    chord = 2 * np.sin(ANG_TOL / 2)
    nbrs = tree.query_ball_point(u, chord, workers=-1)
    for i in range(N):
        idx = nbrs[i]
        if not idx:
            continue
        idx = np.asarray(idx)
        # depth along the ray = projection of acc point onto u_i
        proj = acc_coord[idx] @ u[i]
        behind = idx[(proj > r[i] + MARGIN) & (proj < r[i] + HID_RANGE)]
        if behind.size == 0:
            continue
        hidden_occ[i] = 1
        # nearest-few behind -> class vote
        pb = acc_coord[behind] @ u[i]
        order = behind[np.argsort(pb)][:KNN]
        cls = acc_seg[order]; cls = cls[cls >= 0]
        if cls.size:
            hidden_cls[i] = np.bincount(cls).argmax()
    return hidden_occ, hidden_cls


if __name__ == "__main__":
    # unit-test / sanity on one KITTI scan with +-K sweeps
    import os, sys
    sys.path.insert(0, "/root/WeatherTTA_ARIS8/code")
    from wtta.poses import get_poses, transform_points
    from wtta.labels import map_semantickitti, CLASS_NAMES
    KITTI = "/datasets/data/semiantic_kitti/dataset/sequences/08"
    poses = get_poses("08"); K = 5; i = 1000
    def load(idx):
        p = np.fromfile(f"{KITTI}/velodyne/{idx:06d}.bin", np.float32).reshape(-1, 4)
        s = map_semantickitti(np.fromfile(f"{KITTI}/labels/{idx:06d}.label", np.int32))
        return p[:, :3].astype(np.float32), s
    coord, seg = load(i)
    acc_c, acc_s = [coord], [seg]
    for j in range(i - K, i + K + 1):
        if j == i: continue
        c, s = load(j)
        rel = np.linalg.inv(poses[i]) @ poses[j]
        acc_c.append(transform_points(c.astype(np.float64), rel).astype(np.float32)); acc_s.append(s)
    acc_c = np.concatenate(acc_c, 0); acc_s = np.concatenate(acc_s, 0)
    ho, hc = build_amodal_targets(coord, seg, acc_c, acc_s)
    print(f"scan {i}: N={len(coord)}  hidden_occ frac={ho.mean():.3f}  hidden_cls labeled frac={(hc>=0).mean():.3f}")
    r = np.linalg.norm(coord, axis=1)
    print(f"  hidden_occ frac by range: <10m={ho[r<10].mean():.2f} 10-30m={ho[(r>=10)&(r<30)].mean():.2f} >30m={ho[r>=30].mean():.2f}")
    # how often does the hidden class DIFFER from the visible point's own class (i.e., genuinely new amodal info)?
    valid = (hc >= 0) & (seg >= 0)
    print(f"  hidden-cls != visible-cls (new info) frac among labeled: {(hc[valid]!=seg[valid]).mean():.3f}")
