"""Codex's decisive UPPER-BOUND CHECK (the cheapest go/no-go for the whole amodal-distillation direction):
Does a MULTI-SWEEP teacher (privileged ±K accumulated evidence) beat the SINGLE-SCAN baseline by >=+5pp on
RS-Thing (recoverable-sparse thing points) over full val? If NOT, there is nothing to distill -> STOP.

RS-Thing = current-scan THING points with n1<5 neighbors @0.5m (sparse) AND nK/n1>=4 (recoverable across +-K sweeps).
Both models: same SpUNet backbone, same budget. Teacher input = points accumulated over +-K pose-registered sweeps
(it predicts ALL points incl. current-scan ones, evaluated only on current-scan RS-Thing). Baseline = single scan.
Self-contained; reuses generic tooling (poses, labels). Numbers machine-printed (no hand-typed deltas)."""
import os, sys, glob, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/project/Pointcept"); sys.path.insert(0, "/root/WeatherTTA_ARIS8/code")
from pointcept.models.builder import build_model
from wtta.labels import map_semantickitti
from wtta.poses import get_poses, transform_points
from scipy.spatial import cKDTree

KROOT = "/datasets/data/semiantic_kitti/dataset/sequences"
NUM_CLASSES, IGNORE, GRID, THING = 19, -1, 0.05, list(range(8))
KSWEEP = 5
SP = dict(type="SpUNet-v1m1", in_channels=4, num_classes=NUM_CLASSES,
          channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2))
TRAIN_SEQS = ["00", "01", "02", "03", "09", "10"]
VAL_SEQ = "08"


def load(seq, i):
    p = np.fromfile(f"{KROOT}/{seq}/velodyne/{i:06d}.bin", np.float32).reshape(-1, 4)
    s = map_semantickitti(np.fromfile(f"{KROOT}/{seq}/labels/{i:06d}.label", np.int32))
    return p[:, :3].astype(np.float32), p[:, 3].astype(np.float32), s


def accumulate(seq, i, poses, K=KSWEEP):
    """Return current-scan (coord,strength,seg, mask_current) within the +-K accumulated cloud."""
    c0, st0, s0 = load(seq, i)
    coords, streng, segs = [c0], [st0], [s0]
    n = len(glob.glob(f"{KROOT}/{seq}/velodyne/*.bin"))
    for j in range(i - K, i + K + 1):
        if j == i or j < 0 or j >= n: continue
        try: cj, stj, sj = load(seq, j)
        except Exception: continue
        rel = np.linalg.inv(poses[i]) @ poses[j]
        coords.append(transform_points(cj.astype(np.float64), rel).astype(np.float32)); streng.append(stj); segs.append(sj)
    cur_mask = np.zeros(sum(len(x) for x in coords), bool); cur_mask[:len(c0)] = True
    return (np.concatenate(coords, 0), np.concatenate(streng, 0), np.concatenate(segs, 0), cur_mask)


def vox(coord, strength, aug=False):
    c = coord.astype(np.float32).copy()
    if aug:
        th = np.random.uniform(-np.pi, np.pi); R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
        c = c @ R.T; c *= np.random.uniform(0.95, 1.05)
        if np.random.rand() < 0.5: c[:, 0] = -c[:, 0]
    g = np.floor(c / GRID).astype(np.int64); g -= g.min(0, keepdims=True)
    _, idx, inv = np.unique(g, axis=0, return_index=True, return_inverse=True)
    feat = np.concatenate([c[idx], strength[idx, None]], 1).astype(np.float32)
    return feat, g[idx].astype(np.int32), inv, idx   # idx = first-point index per voxel (use for labels)


def to_in(feat, gc, dev):
    return dict(feat=torch.from_numpy(feat).to(dev), grid_coord=torch.from_numpy(gc).to(dev),
                offset=torch.tensor([len(feat)], dtype=torch.long, device=dev))


def rs_mask(coord, seg, poses_seq, seq, i):
    """recoverable-sparse THING mask on current scan."""
    tree = cKDTree(coord); n1 = tree.query_ball_point(coord, 0.5, return_length=True).astype(np.float32)
    acc, _, _, cur = accumulate(seq, i, poses_seq)
    ta = cKDTree(acc); nK = ta.query_ball_point(coord, 0.5, return_length=True).astype(np.float32)
    rec = (n1 < 5) & (nK / np.maximum(n1, 1) >= 4)
    thing = np.isin(seg, THING) & (seg >= 0)
    return rec & thing


class CM:
    def __init__(s): s.m = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    def add(s, p, g):
        v = (g >= 0) & (g < NUM_CLASSES); p = np.clip(p[v], 0, NUM_CLASSES - 1); g = g[v]
        if len(g): s.m += np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    def miou(s, cls=None):
        I = np.diag(s.m); U = s.m.sum(0) + s.m.sum(1) - I; iou = I / np.maximum(U, 1)
        idx = cls if cls is not None else range(NUM_CLASSES); v = [c for c in idx if U[c] > 0]
        return float(np.mean([iou[c] for c in v])) if v else 0.0


def train(mode, iters, dev, poses_all):
    net = build_model(SP).to(dev); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.005)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.05)
    pool = [(sq, int(os.path.basename(f)[:-4])) for sq in TRAIN_SEQS for f in glob.glob(f"{KROOT}/{sq}/velodyne/*.bin")]
    t0 = time.time()
    for it in range(iters):
        sq, i = pool[np.random.randint(len(pool))]
        try:
            if mode == "teacher":
                coord, st, seg, _ = accumulate(sq, i, poses_all[sq])
            else:
                coord, st, seg = load(sq, i)
        except Exception:
            continue
        feat, gc, inv, idx = vox(coord, st, aug=True)
        logit = net(to_in(feat, gc, dev))
        segv = torch.from_numpy(seg[idx].astype(np.int64)).to(dev)   # label per voxel from SAME unique idx
        loss = F.cross_entropy(logit, segv, ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (it + 1) % 4000 == 0: print(f"  [{mode}] it{it+1} loss={loss.item():.3f} {(time.time()-t0)/(it+1)*1000:.0f}ms/it", flush=True)
    return net


@torch.no_grad()
def eval_rs(net, mode, dev, poses_all, limit=150):
    net.eval(); cm_all, cm_rs = CM(), CM()
    files = sorted(glob.glob(f"{KROOT}/{VAL_SEQ}/velodyne/*.bin"))
    poses_v = poses_all[VAL_SEQ]
    for k in range(0, min(limit * 4, len(files)), 4):
        i = k
        coord, st, seg = load(VAL_SEQ, i)
        rsm = rs_mask(coord, seg, poses_v, VAL_SEQ, i)
        if mode == "teacher":
            acc, ast, aseg, cur = accumulate(VAL_SEQ, i, poses_v)
            feat, gc, inv, _ = vox(acc, ast);
            logit = net(to_in(feat, gc, dev)); predv = logit.argmax(1).cpu().numpy()
            pred_all = predv[inv]; pred = pred_all[cur]   # current-scan points only
        else:
            feat, gc, inv, _ = vox(coord, st); logit = net(to_in(feat, gc, dev)); pred = logit.argmax(1).cpu().numpy()[inv]
        cm_all.add(pred, seg); cm_rs.add(pred[rsm], seg[rsm])
    return cm_all.miou(THING), cm_rs.miou(THING)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--out", default="outputs/ub")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    poses_all = {sq: get_poses(sq) for sq in TRAIN_SEQS + [VAL_SEQ]}
    res = {}
    for mode in ["baseline", "teacher"]:
        print(f"=== train {mode} ===", flush=True)
        net = train(mode, a.iters, dev, poses_all)
        thing, rs = eval_rs(net, mode, dev, poses_all)
        res[mode] = dict(thing_mIoU=round(thing, 4), RS_thing_mIoU=round(rs, 4))
        print(f"[{mode}] thing_mIoU={thing:.4f} RS_thing_mIoU={rs:.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/upperbound.json", "w"), indent=1)
    gap = (res["teacher"]["RS_thing_mIoU"] - res["baseline"]["RS_thing_mIoU"]) * 100
    print(f"\n=== UPPER-BOUND: teacher RS-Thing − baseline = {gap:+.2f} pp (Codex gate: >=+5 to continue) ===", flush=True)
    print(f"==> {'CONTINUE to distillation' if gap >= 5 else 'STOP — no recoverable single-scan signal to distill'}", flush=True)


if __name__ == "__main__":
    main()
