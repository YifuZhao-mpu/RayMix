"""Codex kill-shot KD test (thread 019e8933): the ONE cheap experiment to falsify "amodal distillation is dead".
Fine-tune the CONVERGED single-scan student (B_single) with knowledge distillation from the teacher's ACCUMULATED-
cloud logits (T_accum), evaluated/aligned at the student's single-scan points. In-distribution student (single
scans), teacher only as a soft-label target — so the student never inherits the teacher's OOD-on-sparse weights.

L = CE(hard) + 0.5 * T^2 * KL(softmax(T_accum/T) || softmax(student/T)),  T=2.
Short schedule (~0.2x: 10k iters), 1 seed, same aug/opt as baseline. Start from B_single ckpt.
GO iff RS-Thing >= +2.5pp over the matched baseline AND overall val mIoU not down >0.5pp. Else amodal DEAD (honest).
All numbers machine-printed via converged_upperbound-style eval at the end.
"""
import os, sys, glob, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/project/Pointcept")
from pointcept.models.builder import build_model
from pointcept.datasets.semantic_kitti_multisweep import _load_poses, MOVING_RAW

KROOT = "/datasets/data/semiantic_kitti/dataset/sequences"
NUM_CLASSES, IGNORE, GRID, THING, KSWEEP = 19, -1, 0.05, list(range(8)), 5
TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL = "08"
SP = dict(type="SpUNet-v1m1", in_channels=4, num_classes=NUM_CLASSES,
          channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2))
LMAP = None
BASE_CKPT = "/root/project/Pointcept/exp/semantic_kitti/fresh_amodal_baseline_spunet/model/model_best.pth"
TEACH_CKPT = "/root/project/Pointcept/exp/semantic_kitti/fresh_amodal_teacher_spunet/model/model_best.pth"


def learning_map():
    from pointcept.datasets.semantic_kitti import SemanticKITTIDataset
    return SemanticKITTIDataset.get_learning_map(IGNORE)


def read_scan(seq, i, want_moving=False):
    p = np.fromfile(f"{KROOT}/{seq}/velodyne/{i:06d}.bin", np.float32).reshape(-1, 4)
    coord, strength = p[:, :3].astype(np.float32), p[:, 3].astype(np.float32)
    lp = f"{KROOT}/{seq}/labels/{i:06d}.label"
    if os.path.exists(lp):
        raw = np.fromfile(lp, np.int32).reshape(-1) & 0xFFFF
        seg = np.vectorize(LMAP.__getitem__)(raw).astype(np.int64)
        mov = np.isin(raw, list(MOVING_RAW)) if want_moving else None
    else:
        seg = np.zeros(len(coord), np.int64); mov = None
    return coord, strength, seg, mov


def accumulate(seq, i, poses):
    c0, s0, _, _ = read_scan(seq, i)
    coords, strs = [c0], [s0]; n = len(glob.glob(f"{KROOT}/{seq}/velodyne/*.bin"))
    Ti = np.linalg.inv(poses[i])
    for j in range(i - KSWEEP, i + KSWEEP + 1):
        if j == i or j < 0 or j >= n: continue
        cj, sj, _, movj = read_scan(seq, j, want_moving=True)
        if movj is not None:
            keep = ~movj; cj, sj = cj[keep], sj[keep]
        rel = Ti @ poses[j]
        cj_i = (rel @ np.hstack([cj, np.ones((len(cj), 1))]).T).T[:, :3].astype(np.float32)
        coords.append(cj_i); strs.append(sj)
    cur = np.zeros(sum(len(x) for x in coords), bool); cur[:len(c0)] = True
    return np.concatenate(coords, 0).astype(np.float32), np.concatenate(strs, 0).astype(np.float32), cur


def voxelize(coord, strength, aug=False):
    c = coord.astype(np.float32).copy()
    if aug:
        th = np.random.uniform(-np.pi, np.pi); R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
        c = c @ R.T; c *= np.random.uniform(0.95, 1.05)
        if np.random.rand() < 0.5: c[:, 0] = -c[:, 0]
    g = np.floor(c / GRID).astype(np.int64); g -= g.min(0, keepdims=True)
    _, idx, inv = np.unique(g, axis=0, return_index=True, return_inverse=True)
    feat = np.concatenate([c[idx], strength[idx, None]], 1).astype(np.float32)
    return feat, g[idx].astype(np.int32), inv, idx, c


def to_in(feat, gc, dev):
    return dict(feat=torch.from_numpy(feat).to(dev), grid_coord=torch.from_numpy(gc).to(dev),
                offset=torch.tensor([len(feat)], dtype=torch.long, device=dev))


def load_ckpt(path, dev):
    net = build_model(SP).to(dev)
    sd = torch.load(path, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
    sd2 = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."): kk = kk[7:]
        if kk.startswith("backbone."): kk = kk[9:]
        sd2[kk] = v
    net.load_state_dict(sd2, strict=False); return net


class CM:
    def __init__(s): s.m = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    def add(s, p, g):
        v = (g >= 0) & (g < NUM_CLASSES); p = np.clip(p[v], 0, NUM_CLASSES - 1); g = g[v]
        if len(g): s.m += np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    def miou(s, cls=None):
        I = np.diag(s.m); U = s.m.sum(0) + s.m.sum(1) - I; iou = I / np.maximum(U, 1)
        idx = cls if cls is not None else range(NUM_CLASSES); v = [c for c in idx if U[c] > 0]
        return float(np.mean([iou[c] for c in v])) if v else 0.0


@torch.no_grad()
def evaluate(student, dev, poses_val, limit=400):
    cm_all, cm_rs = CM(), CM()
    files = sorted(glob.glob(f"{KROOT}/{VAL}/velodyne/*.bin"))
    from scipy.spatial import cKDTree
    for k in range(0, min(limit * 4, len(files)), 4):
        i = k; coord, st, seg, _ = read_scan(VAL, i)
        t1 = cKDTree(coord); n1 = t1.query_ball_point(coord, 0.5, return_length=True).astype(np.float32)
        acc, _, _ = accumulate(VAL, i, poses_val)
        ta = cKDTree(acc); nK = ta.query_ball_point(coord, 0.5, return_length=True).astype(np.float32)
        rs = np.isin(seg, THING) & (seg >= 0) & (n1 < 5) & (nK / np.maximum(n1, 1) >= 4) & (nK >= 8)
        f, g, inv, _, _ = voxelize(coord, st)
        pred = student(to_in(f, g, dev)).argmax(1).cpu().numpy()[inv]
        cm_all.add(pred, seg); cm_rs.add(pred[rs], seg[rs])
    return round(cm_all.miou(), 4), round(cm_rs.miou(THING), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000); ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="/root/Fresh_ARIS8/code/outputs/killshot/kd.json")
    a = ap.parse_args()
    global LMAP; LMAP = learning_map()
    dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    teacher = load_ckpt(TEACH_CKPT, dev); teacher.eval()
    student = load_ckpt(BASE_CKPT, dev); student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=4e-4, weight_decay=0.005)   # low lr: fine-tune from converged
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4, total_steps=a.iters, pct_start=0.05)
    poses = {sq: _load_poses(f"{KROOT}/{sq}") for sq in TRAIN_SEQS + [VAL]}
    pool = [(sq, int(os.path.basename(f)[:-4])) for sq in TRAIN_SEQS for f in glob.glob(f"{KROOT}/{sq}/velodyne/*.bin")]
    print(f"[killshot KD] start from B_single; {a.iters} iters; T={a.T} alpha={a.alpha}", flush=True)
    t0 = time.time()
    for it in range(a.iters):
        sq, i = pool[np.random.randint(len(pool))]
        try:
            coord, st, seg, _ = read_scan(sq, i)
            acc, ast, cur = accumulate(sq, i, poses[sq])
        except Exception:
            continue
        # student voxelization (single scan, no aug so teacher logits align to same current points)
        fs, gs, invs, idxs, _ = voxelize(coord, st, aug=False)
        seg_v = torch.from_numpy(seg[idxs].astype(np.int64)).to(dev)
        # teacher logits on accumulated cloud, read at the SAME current points, then to student voxels via idxs
        with torch.no_grad():
            fa, ga, inva, _, _ = voxelize(acc, ast, aug=False)
            t_logit_pts = teacher(to_in(fa, ga, dev))[inva][torch.from_numpy(cur).to(dev)]  # current-scan points
            t_logit_v = t_logit_pts[torch.from_numpy(idxs).to(dev)]                          # at student voxels
        s_logit = student(to_in(fs, gs, dev))
        ce = F.cross_entropy(s_logit, seg_v, ignore_index=IGNORE)
        kd = F.kl_div(F.log_softmax(s_logit / a.T, 1), F.softmax(t_logit_v / a.T, 1), reduction="batchmean") * (a.T ** 2)
        loss = ce + a.alpha * kd
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (it + 1) % 2000 == 0:
            print(f"  it{it+1} ce={ce.item():.3f} kd={kd.item():.3f} {(time.time()-t0)/(it+1)*1000:.0f}ms/it", flush=True)
    student.eval()
    kd_all, kd_rs = evaluate(student, dev, poses[VAL])
    # matched baseline numbers (the converged B_single) — recompute with same eval for a clean paired comparison
    base = load_ckpt(BASE_CKPT, dev); base.eval()
    b_all, b_rs = evaluate(base, dev, poses[VAL])
    res = dict(baseline_overall=b_all, baseline_RS_thing=b_rs, kd_overall=kd_all, kd_RS_thing=kd_rs,
               d_RS_thing_pp=round((kd_rs - b_rs) * 100, 2), d_overall_pp=round((kd_all - b_all) * 100, 2))
    res["GO"] = bool(res["d_RS_thing_pp"] >= 2.5 and res["d_overall_pp"] >= -0.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f"\n=== KILL-SHOT KD: RS-Thing {res['d_RS_thing_pp']:+.2f}pp (gate +2.5), overall {res['d_overall_pp']:+.2f}pp "
          f"(floor -0.5) => {'GO — distillation transfers, PROCEED Phase 2' if res['GO'] else 'NO-GO — amodal DEAD (honest), single-scan caps ~7'}", flush=True)


if __name__ == "__main__":
    main()
