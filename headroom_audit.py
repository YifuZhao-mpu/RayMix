"""Pilot 0: amodal headroom audit for single-scan LiDAR seg.
Hypothesis: the points a single-scan baseline gets WRONG (esp. thing-class / far) sit in regions that are
SPARSE in the single scan but become densely OBSERVED across +-5 pose-registered sweeps -> there is hidden
amodal context to exploit. Measures: (a) error rate vs single-scan local density, (b) densification (d_acc/d_single)
on wrong thing/far points. KILL the direction if wrong-thing points do NOT gain substantial cross-sweep context.
Reuses generic tooling (poses, SpUNet inference, labels) from the prior project — vanilla utilities, not results."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/root/WeatherTTA_ARIS8/code")
from wtta.model import SourceModel, voxelize
from wtta.poses import get_poses, transform_points
from wtta.labels import map_semantickitti
from scipy.spatial import cKDTree

KITTI = "/datasets/data/semiantic_kitti/dataset/sequences/08"
CKPT = "/root/project/Pointcept/exp/semantic_kitti/weathertta_src_spunet/model/model_best.pth"
THING = set(range(8))   # SemanticKITTI train ids 0..7 = car,bicycle,motorcycle,truck,other-veh,person,bicyclist,motorcyclist
R = 0.5                 # local-density radius (m)
NACC = 5                # +-5 sweeps

def load_scan(seq_dir, idx):
    v = f"{seq_dir}/velodyne/{idx:06d}.bin"
    if not os.path.exists(v): return None
    pts = np.fromfile(v, dtype=np.float32).reshape(-1, 4)
    lab = f"{seq_dir}/labels/{idx:06d}.label"
    seg = map_semantickitti(np.fromfile(lab, dtype=np.int32)) if os.path.exists(lab) else None
    return pts[:, :3].astype(np.float32), pts[:, 3].astype(np.float32), seg

def main(n=120, stride=10):
    src = SourceModel(CKPT)
    poses = get_poses("08")
    nframe = len(poses)
    dens_gain_thing, dens_gain_far = [], []
    err_by_dens = {b: [0, 0] for b in range(6)}   # bin by single-scan density -> [wrong, total]
    visible_thing, visible_far = [], []
    nscan = 0
    for i in range(20, nframe - 20, stride):
        if nscan >= n: break
        cur = load_scan(KITTI, i)
        if cur is None or cur[2] is None: continue
        coord, strength, seg = cur
        valid = seg >= 0
        # single-scan baseline prediction
        vox, inv = voxelize(coord, strength)
        with torch.no_grad():
            pred = src.backbone(src._to_input(vox)).argmax(1).cpu().numpy()[inv]
        wrong = (pred != seg) & valid
        # accumulate +-NACC sweeps into frame i coords
        acc = [coord]
        for j in range(i - NACC, i + NACC + 1):
            if j == i or j < 0 or j >= nframe: continue
            sc = load_scan(KITTI, j)
            if sc is None: continue
            rel = np.linalg.inv(poses[i]) @ poses[j]
            acc.append(transform_points(sc[0].astype(np.float64), rel).astype(np.float32))
        acc = np.concatenate(acc, 0)
        ts, ta = cKDTree(coord), cKDTree(acc)
        d_single = ts.query_ball_point(coord, R, return_length=True).astype(np.float32)
        d_acc = ta.query_ball_point(coord, R, return_length=True).astype(np.float32)
        rng = np.linalg.norm(coord, axis=1)
        # error rate vs single-scan density
        for b, (lo, hi) in enumerate([(0,5),(5,10),(10,20),(20,40),(40,80),(80,1e9)]):
            m = valid & (d_single >= lo) & (d_single < hi)
            err_by_dens[b][0] += int((wrong & m).sum()); err_by_dens[b][1] += int(m.sum())
        gain = d_acc / np.maximum(d_single, 1)
        wt = wrong & np.isin(seg, list(THING))
        wf = wrong & (rng > 30)
        if wt.sum(): dens_gain_thing.append(float(np.median(gain[wt]))); visible_thing.append(float((gain[wt] >= 2).mean()))
        if wf.sum(): dens_gain_far.append(float(np.median(gain[wf]))); visible_far.append(float((gain[wf] >= 2).mean()))
        nscan += 1
    print(f"\n=== AMODAL HEADROOM AUDIT ({nscan} KITTI val scans, +-{NACC} sweeps, R={R}m) ===")
    print("error rate vs single-scan local density (neighbors within R):")
    for b, name in zip(range(6), ["0-5","5-10","10-20","20-40","40-80","80+"]):
        w, t = err_by_dens[b]
        print(f"  density {name:6s}: err={w/max(t,1)*100:5.1f}%  (n={t})")
    print(f"\nWRONG THING points: median densification d_acc/d_single = {np.mean(dens_gain_thing):.2f}x ; "
          f"frac with >=2x context gain = {np.mean(visible_thing)*100:.1f}%")
    print(f"WRONG FAR(>30m) points: median densification = {np.mean(dens_gain_far):.2f}x ; "
          f"frac with >=2x context gain = {np.mean(visible_far)*100:.1f}%")
    print(f"\nDECISION: amodal headroom EXISTS if wrong-thing >=2x-gain frac >= 15% AND errors concentrate at low density.")

if __name__ == "__main__":
    main()
