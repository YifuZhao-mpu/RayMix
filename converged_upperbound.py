"""CONVERGED upper-bound re-test (Phase 1c) — HARDENED per Codex review (thread 019e826a).
Decisive gate for the amodal direction, with the distillation-transferability split.

Three readouts on the SAME val frames / SAME current-point masks:
  B_single : baseline ckpt on single scan
  T_single : teacher ckpt on SINGLE scan          (<- the control Codex required)
  T_accum  : teacher ckpt on ±K accumulated cloud, read out at current-scan points
Decompose RS-Thing:
  T_accum - B_single  = total privileged headroom
  T_accum - T_single  = needs inference-time extra evidence (distillation CANNOT transfer)
  T_single - B_single = train/prior effect (distillation CAN transfer)  <- distillation-relevant
Stratify by current-support bins {0-1,2-4,>=5}, range, thing-class. Paired bootstrap over FRAMES.
Pre-registered gate frozen in AMODAL_MULTIWEEK_PLAN.md. All numbers machine-printed.
"""
import os, sys, glob, argparse, json, numpy as np, torch
sys.path.insert(0, "/root/project/Pointcept")
from pointcept.models.builder import build_model
from pointcept.datasets.semantic_kitti_multisweep import _load_poses, MOVING_RAW
from scipy.spatial import cKDTree

KROOT = "/datasets/data/semiantic_kitti/dataset/sequences"
NUM_CLASSES, IGNORE, GRID, THING, KSWEEP, VAL = 19, -1, 0.05, list(range(8)), 5, "08"
SP = dict(type="SpUNet-v1m1", in_channels=4, num_classes=NUM_CLASSES,
          channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2))
LMAP = None
RADII = (0.3, 0.5, 0.7)
MAIN_R = 0.5


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


def accumulate_static(seq, i, poses):
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


def voxelize(coord, strength):
    g = np.floor(coord / GRID).astype(np.int64); g -= g.min(0, keepdims=True)
    _, idx, inv = np.unique(g, axis=0, return_index=True, return_inverse=True)
    feat = np.concatenate([coord[idx], strength[idx, None]], 1).astype(np.float32)
    return feat, g[idx].astype(np.int32), inv


def to_in(feat, gc, dev):
    return dict(feat=torch.from_numpy(feat).to(dev), grid_coord=torch.from_numpy(gc).to(dev),
                offset=torch.tensor([len(feat)], dtype=torch.long, device=dev))


def load_ckpt(path, dev):
    net = build_model(SP).to(dev)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd2 = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."): kk = kk[7:]
        if kk.startswith("backbone."): kk = kk[9:]
        sd2[kk] = v
    m, u = net.load_state_dict(sd2, strict=False)
    print(f"  loaded {os.path.basename(path)}: missing {len(m)} unexpected {len(u)}", flush=True)
    net.eval(); return net


@torch.no_grad()
def predict_single(net, coord, st, dev):
    f, g, inv = voxelize(coord, st)
    return net(to_in(f, g, dev)).argmax(1).cpu().numpy()[inv]


@torch.no_grad()
def predict_accum(net, acc, ast, cur, dev):
    f, g, inv = voxelize(acc, ast)
    return net(to_in(f, g, dev)).argmax(1).cpu().numpy()[inv][cur]


def per_frame_iou_accumulate(store, pred, seg, mask):
    """accumulate intersection/union counts per class for points selected by mask (for pooled mIoU)."""
    p, g = pred[mask], seg[mask]
    v = (g >= 0) & (g < NUM_CLASSES); p, g = np.clip(p[v], 0, NUM_CLASSES - 1), g[v]
    if len(g):
        store += np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def miou_from_cm(cm, classes):
    I = np.diag(cm); U = cm.sum(0) + cm.sum(1) - I; iou = I / np.maximum(U, 1)
    v = [c for c in classes if U[c] > 0]
    return float(np.mean([iou[c] for c in v])) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="/root/project/Pointcept/exp/semantic_kitti/fresh_amodal_baseline_spunet/model/model_best.pth")
    ap.add_argument("--teacher", default="/root/project/Pointcept/exp/semantic_kitti/fresh_amodal_teacher_spunet/model/model_best.pth")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default="/root/Fresh_ARIS8/code/outputs/ub_converged/upperbound.json")
    a = ap.parse_args()
    global LMAP; LMAP = learning_map()
    dev = "cuda"
    base = load_ckpt(a.baseline, dev); teach = load_ckpt(a.teacher, dev)
    poses = _load_poses(f"{KROOT}/{VAL}")
    files = sorted(glob.glob(f"{KROOT}/{VAL}/velodyne/*.bin"))
    idxs = list(range(0, min(a.limit * 4, len(files)), 4))

    # confusion stores: [readout][slice] ; slices: all, thing, rs(main), and support bins of rs
    readouts = ["B_single", "T_single", "T_accum"]
    slices = ["all", "thing", "rs", "rs_sup01", "rs_sup24", "rs_sup5p", "rs_near", "rs_far"]
    cm = {r: {s: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for s in slices} for r in readouts}
    # per-frame RS-Thing mIoU for paired bootstrap
    per_frame = {r: [] for r in readouts}
    rs_count = 0

    for i in idxs:
        coord, st, seg, mov = read_scan(VAL, i, want_moving=True)
        acc, ast, cur = accumulate_static(VAL, i, poses)
        t1 = cKDTree(coord); n1 = t1.query_ball_point(coord, MAIN_R, return_length=True).astype(np.float32)
        ta = cKDTree(acc); nK = ta.query_ball_point(coord, MAIN_R, return_length=True).astype(np.float32)
        rng = np.linalg.norm(coord, axis=1)
        thing = np.isin(seg, THING) & (seg >= 0)
        # recoverable-sparse: sparse in single AND >=4x recovered AND absolute accum floor (Codex)
        rs = thing & (n1 < 5) & (nK / np.maximum(n1, 1) >= 4) & (nK >= 8)
        rs_count += int(rs.sum())
        masks = dict(all=(seg >= 0), thing=thing, rs=rs,
                     rs_sup01=rs & (n1 <= 1), rs_sup24=rs & (n1 >= 2) & (n1 <= 4), rs_sup5p=rs & (n1 >= 5),
                     rs_near=rs & (rng <= 20), rs_far=rs & (rng > 20))
        preds = dict(B_single=predict_single(base, coord, st, dev),
                     T_single=predict_single(teach, coord, st, dev),
                     T_accum=predict_accum(teach, acc, ast, cur, dev))
        for r in readouts:
            for s in slices:
                per_frame_iou_accumulate(cm[r][s], preds[r], seg, masks[s])
            # per-frame RS-Thing (thing-class mIoU on rs of this frame) for bootstrap
            fcm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
            per_frame_iou_accumulate(fcm, preds[r], seg, masks["rs"])
            per_frame[r].append(miou_from_cm(fcm, THING))

    def M(r, s): return round(miou_from_cm(cm[r][s], THING if s != "all" else range(NUM_CLASSES)), 4)
    res = {r: {s: M(r, s) for s in slices} for r in readouts}
    res["rs_thing_count"] = rs_count
    res["splits_pp"] = dict(
        total_headroom=round((res["T_accum"]["rs"] - res["B_single"]["rs"]) * 100, 2),
        needs_inference_evidence=round((res["T_accum"]["rs"] - res["T_single"]["rs"]) * 100, 2),
        distillation_transferable=round((res["T_single"]["rs"] - res["B_single"]["rs"]) * 100, 2),
    )
    # paired bootstrap over frames on the distillation-transferable split (T_single - B_single)
    bt, bb = np.array(per_frame["T_single"]), np.array(per_frame["B_single"])
    diff = bt - bb; n = len(diff); rng = np.random.RandomState(0)
    boots = np.array([diff[rng.randint(0, n, n)].mean() for _ in range(5000)]) * 100
    res["distill_transferable_CI95_pp"] = [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]
    # gate (pre-registered, frozen): total >=5pp AND broad (2-4 bin positive) AND some transferable share
    s = res["splits_pp"]
    bin24_gain = round((res["T_accum"]["rs_sup24"] - res["B_single"]["rs_sup24"]) * 100, 2)
    res["bin24_total_gain_pp"] = bin24_gain
    res["GATE"] = ("PROCEED" if (s["total_headroom"] >= 5 and bin24_gain > 0 and s["distillation_transferable"] >= 1.0)
                   else ("AMBIGUOUS" if s["total_headroom"] >= 5 else "STOP"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f"\n=== RS-Thing splits (pp): total {s['total_headroom']:+.2f} | needs-inference-evidence "
          f"{s['needs_inference_evidence']:+.2f} | DISTILL-TRANSFERABLE {s['distillation_transferable']:+.2f} "
          f"CI95{res['distill_transferable_CI95_pp']} | 2-4bin {bin24_gain:+.2f} | n_rs={rs_count} => {res['GATE']}", flush=True)


if __name__ == "__main__":
    main()
