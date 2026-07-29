"""Pilot-1: does the amodal aux head improve single-scan LiDAR seg on HARD slices?
Self-contained (my voxelize controls point->voxel alignment; Pointcept SpUNet backbone as feature extractor).
Controlled: --aux 0/1 toggles amodal aux loss; --fuse 0/1 toggles feeding predicted hidden context back into
features; --beam_aug 0/1 toggles train-time beam-dropout (fixes train-dense/test-sparse mismatch). Decisive gate
(Codex): keep iff +1.5 mIoU on 16-beam val OR +2.5 thing-mIoU on >30m slice (with >=+0.5 overall).
"""
import os, sys, glob, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/project/Pointcept")
sys.path.insert(0, "/root/WeatherTTA_ARIS8/code")
from pointcept.models.builder import build_model
from wtta.labels import map_semantickitti

KROOT = "/datasets/data/semiantic_kitti/dataset/sequences"
CACHE = "/root/Fresh_ARIS8/outputs/amodal_cache"
NUM_CLASSES, IGNORE = 19, -1
THING = list(range(8))
GRID = 0.05
SPUNET_FEAT = dict(type="SpUNet-v1m1", in_channels=4, num_classes=0,   # num_classes=0 -> returns FEATURES
                   channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2))


class AmodalNet(nn.Module):
    def __init__(self, feat_dim=96, fuse=False):
        super().__init__()
        self.backbone = build_model(SPUNET_FEAT)
        self.fuse = fuse
        self.occ = nn.Linear(feat_dim, 2)              # hidden-occupied behind? (geometric densification aux)
        self.hcls = nn.Linear(feat_dim, NUM_CLASSES)   # hidden class behind (boundary-weighted)
        if fuse:                                        # FUSION: feed predicted amodal context back into feats
            self.refine = nn.Sequential(nn.Linear(feat_dim + 2 + NUM_CLASSES, feat_dim),
                                        nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True))
        self.seg = nn.Linear(feat_dim, NUM_CLASSES)

    def forward(self, inp):
        f = self.backbone(inp)
        occ, hcls = self.occ(f), self.hcls(f)
        if self.fuse:
            ctx = torch.cat([f, F.softmax(occ, 1), F.softmax(hcls, 1)], 1)
            f = self.refine(ctx)                        # refined features carry the predicted hidden context
        return self.seg(f), occ, hcls


def beam_dropout(coord, p_keep):
    """Simulate a lower-beam sensor by keeping a fraction of the 64 laser rings (by pitch). Forces the model
    (and the amodal fusion head) to see SPARSE inputs at train time — exactly where amodal context should help."""
    r = np.linalg.norm(coord, axis=1) + 1e-6
    pitch = np.arcsin(np.clip(coord[:, 2] / r, -1, 1))
    ring = np.clip(((1 - (pitch + 0.4363) / 0.4712) * 64).astype(int), 0, 63)
    keep_rings = np.random.rand(64) < p_keep
    return keep_rings[ring]


def voxelize_aug(coord, strength, seg, ho, hc, aug=True, beam_aug=False):
    c = coord.astype(np.float32).copy()
    keep = np.ones(len(c), bool)
    if aug:
        th = np.random.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
        c = c @ R.T
        c *= np.random.uniform(0.95, 1.05)
        if np.random.rand() < 0.5:
            c[:, 0] = -c[:, 0]
    if beam_aug and np.random.rand() < 0.5:             # 50% of scans -> simulate a sparser sensor
        keep = beam_dropout(coord, p_keep=np.random.choice([0.25, 0.5]))   # ~16 or ~32 beam
        if keep.sum() < 1000:
            keep = np.ones(len(c), bool)
    c, strength, seg, ho, hc = c[keep], strength[keep], seg[keep], ho[keep], hc[keep]
    g = np.floor(c / GRID).astype(np.int64); g -= g.min(0, keepdims=True)
    _, idx_first, inverse = np.unique(g, axis=0, return_index=True, return_inverse=True)
    feat = np.concatenate([c[idx_first], strength[idx_first, None]], 1).astype(np.float32)
    gc = g[idx_first].astype(np.int32)
    seg_v = seg[idx_first]
    hc_v = hc[idx_first]
    occ_v = np.zeros(len(idx_first), np.int64)
    np.maximum.at(occ_v, inverse, ho.astype(np.int64))
    return feat, gc, seg_v.astype(np.int64), occ_v, hc_v.astype(np.int64)


def to_input(feat, gc, dev):
    return dict(feat=torch.from_numpy(feat).to(dev), grid_coord=torch.from_numpy(gc).to(dev),
                offset=torch.tensor([len(feat)], dtype=torch.long, device=dev))


def load_raw(seq, i):
    p = np.fromfile(f"{KROOT}/{seq}/velodyne/{i:06d}.bin", np.float32).reshape(-1, 4)
    s = map_semantickitti(np.fromfile(f"{KROOT}/{seq}/labels/{i:06d}.label", np.int32))
    return p[:, :3].astype(np.float32), p[:, 3].astype(np.float32), s


class ConfMat:
    def __init__(s): s.m = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    def add(s, p, g):
        v = (g >= 0) & (g < NUM_CLASSES); p = np.clip(p[v], 0, NUM_CLASSES - 1); g = g[v]
        s.m += np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    def miou(s, classes=None):
        I = np.diag(s.m); U = s.m.sum(0) + s.m.sum(1) - I
        iou = I / np.maximum(U, 1); idx = classes if classes is not None else range(NUM_CLASSES)
        valid = [c for c in idx if U[c] > 0]; return float(np.mean([iou[c] for c in valid])) if valid else 0.0


def evaluate(net, dev, val_seq="08", limit=120, beam16=False):
    net.eval(); cm, cm_far_thing = ConfMat(), ConfMat()
    files = sorted(glob.glob(f"{KROOT}/{val_seq}/velodyne/*.bin"))
    for k in range(0, min(limit * 4, len(files)), 4):
        i = k
        coord, strength, seg = load_raw(val_seq, i)
        if beam16:
            r = np.linalg.norm(coord, axis=1) + 1e-6; pitch = np.arcsin(np.clip(coord[:, 2] / r, -1, 1))
            ring = np.clip(((1 - (pitch + 0.4363) / 0.4712) * 64).astype(int), 0, 63)
            m = (ring % 4 == 0); coord, strength, seg = coord[m], strength[m], seg[m]
        feat, gc, _, _, _ = voxelize_aug(coord, strength, seg, np.zeros(len(seg)), np.zeros(len(seg), np.int8), aug=False)
        g2 = np.floor(coord / GRID).astype(np.int64); g2 -= g2.min(0, keepdims=True)
        _, _, inv = np.unique(g2, axis=0, return_index=True, return_inverse=True)
        with torch.no_grad():
            logit, _, _ = net(to_input(feat, gc, dev))
        pred = logit.argmax(1).cpu().numpy()[inv]
        cm.add(pred, seg)
        rng = np.linalg.norm(coord, axis=1); far = rng > 30
        cm_far_thing.add(pred[far], seg[far])
    return cm.miou(), cm.miou(THING), cm_far_thing.miou(THING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", type=int, default=1)
    ap.add_argument("--fuse", type=int, default=0)
    ap.add_argument("--beam_aug", type=int, default=0)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lo", type=float, default=0.3); ap.add_argument("--lc", type=float, default=0.3)
    ap.add_argument("--seqs", nargs="+", default=["00", "01", "02"])
    ap.add_argument("--out", default="outputs/pilot")
    args = ap.parse_args()
    dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    net = AmodalNet(fuse=bool(args.fuse)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.005)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.iters, pct_start=0.05)
    pool = []
    for sq in args.seqs:
        for f in sorted(glob.glob(f"{CACHE}/{sq}/*.npz")):
            pool.append((sq, int(os.path.basename(f)[:-4])))
    print(f"[pilot aux={args.aux} fuse={args.fuse} beam_aug={args.beam_aug}] pool {len(pool)}; {args.iters} iters", flush=True)
    net.train(); t0 = time.time()
    for it in range(args.iters):
        sq, i = pool[np.random.randint(len(pool))]
        coord, strength, seg = load_raw(sq, i)
        d = np.load(f"{CACHE}/{sq}/{i:06d}.npz"); ho, hc = d["hidden_occ"], d["hidden_cls"].astype(np.int64)
        feat, gc, seg_v, occ_v, hc_v = voxelize_aug(coord, strength, seg, ho, hc, beam_aug=bool(args.beam_aug))
        logit, occ, hcls = net(to_input(feat, gc, dev))
        seg_t = torch.from_numpy(seg_v).to(dev)
        loss = F.cross_entropy(logit, seg_t, ignore_index=IGNORE)
        if args.aux:
            occ_t = torch.from_numpy(occ_v).to(dev)
            loss = loss + args.lo * F.cross_entropy(occ, occ_t)
            hcm = (hc_v >= 0)
            if hcm.sum() > 10:
                w = np.where(hc_v[hcm] != seg_v[hcm], 3.0, 1.0).astype(np.float32)
                hl = F.cross_entropy(hcls[hcm], torch.from_numpy(hc_v[hcm]).to(dev), reduction="none")
                loss = loss + args.lc * (hl * torch.from_numpy(w).to(dev)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % 2000 == 0:
            print(f"  it{it+1} loss={loss.item():.3f} ({(time.time()-t0)/(it+1)*1000:.0f}ms/it)", flush=True)
    miou, tmiou, far_t = evaluate(net, dev); miou16, _, _ = evaluate(net, dev, beam16=True)
    res = dict(aux=args.aux, fuse=args.fuse, beam_aug=args.beam_aug, val_mIoU=round(miou, 4),
               val_thing_mIoU=round(tmiou, 4), far30_thing_mIoU=round(far_t, 4), beam16_mIoU=round(miou16, 4))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(f"{args.out}/pilot_a{args.aux}_f{args.fuse}_b{args.beam_aug}.json", "w"), indent=1)
    print(f"[RESULT aux={args.aux} fuse={args.fuse} beam_aug={args.beam_aug}] {res}", flush=True)


if __name__ == "__main__":
    main()
