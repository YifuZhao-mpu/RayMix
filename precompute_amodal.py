"""Precompute + cache amodal targets (hidden_occ, hidden_cls) per training scan for Pilot-1.
Runs the verified generator over a KITTI-train subset; caches .npz per scan. Boundary weight (hidden_cls!=seg)
is computed at train time from the cached arrays. Low-risk (just the verified generator over scans)."""
import os, sys, glob, time, numpy as np
sys.path.insert(0, "/root/WeatherTTA_ARIS8/code")
sys.path.insert(0, "/root/Fresh_ARIS8/code")
from wtta.poses import get_poses, transform_points
from wtta.labels import map_semantickitti
from amodal_supervision import build_amodal_targets

ROOT = "/datasets/data/semiantic_kitti/dataset/sequences"
OUT = "/root/Fresh_ARIS8/outputs/amodal_cache"
K = 5

def load(seq_dir, idx):
    p = np.fromfile(f"{seq_dir}/velodyne/{idx:06d}.bin", np.float32).reshape(-1, 4)
    s = map_semantickitti(np.fromfile(f"{seq_dir}/labels/{idx:06d}.label", np.int32))
    return p[:, :3].astype(np.float32), s

def main(seqs=("00",), stride=2):
    for sq in seqs:
        seq_dir = f"{ROOT}/{sq}"
        poses = get_poses(sq)
        n = len(glob.glob(f"{seq_dir}/velodyne/*.bin"))
        os.makedirs(f"{OUT}/{sq}", exist_ok=True)
        t0 = time.time(); done = 0
        for i in range(K, n - K, stride):
            cpath = f"{OUT}/{sq}/{i:06d}.npz"
            if os.path.exists(cpath): continue
            try:
                coord, seg = load(seq_dir, i)
                acc_c, acc_s = [coord], [seg]
                for j in range(i - K, i + K + 1):
                    if j == i: continue
                    c, s = load(seq_dir, j)
                    rel = np.linalg.inv(poses[i]) @ poses[j]
                    acc_c.append(transform_points(c.astype(np.float64), rel).astype(np.float32)); acc_s.append(s)
                ho, hc = build_amodal_targets(coord, seg, np.concatenate(acc_c, 0), np.concatenate(acc_s, 0))
                np.savez(cpath, hidden_occ=ho.astype(np.uint8), hidden_cls=hc.astype(np.int8))
                done += 1
                if done % 100 == 0:
                    print(f"  seq{sq} {done} done ({(time.time()-t0)/done:.2f}s/scan)", flush=True)
            except Exception as e:
                print(f"  skip {sq}/{i}: {e}", flush=True)
        print(f"seq{sq}: cached {done} scans in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    import sys
    seqs = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("00", "01", "02")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    main(seqs=seqs, stride=stride)
