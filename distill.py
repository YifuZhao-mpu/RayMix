"""Decisive distillation test: can a single-scan STUDENT recover a meaningful fraction of the teacher's +21.69pp
RS-Thing advantage at ZERO test cost? (Codex's recoverability-gated sweep distillation.)
Teacher: SpUNet on ±K accumulated sweeps (privileged). Student: SpUNet on single scan.
Loss = CE(y, S) + lambda * mean_i[ w_i * KL(softmax(T_i/tau) || softmax(S_i/tau)) ], where the teacher posterior
T_i is taken at the CURRENT-scan points, and w_i gates to RECOVERABLE-SPARSE points (n1<5@0.5m & nK/n1>=4) x teacher conf.
Controlled: --distill 0 (baseline) vs 1. 3 seeds. Stable metric RS-Thing (full val). Numbers machine-printed.
"""
import os, sys, glob, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/project/Pointcept"); sys.path.insert(0, "/root/WeatherTTA_ARIS8/code")
sys.path.insert(0, "/root/Fresh_ARIS8/code")
from pointcept.models.builder import build_model
from teacher_upperbound import (load, accumulate, vox, to_in, rs_mask, CM, train as train_plain,
                                KROOT, NUM_CLASSES, IGNORE, GRID, THING, KSWEEP, SP, TRAIN_SEQS, VAL_SEQ)
from wtta.poses import get_poses
from scipy.spatial import cKDTree


def build_teacher(iters, dev, poses_all, ckpt="outputs/ub_teacher.pt"):
    if os.path.exists(ckpt):
        net = build_model(SP).to(dev); net.load_state_dict(torch.load(ckpt, map_location=dev)); net.eval()
        print("loaded cached teacher", flush=True); return net
    print("training teacher for distillation...", flush=True)
    net = train_plain("teacher", iters, dev, poses_all); net.eval()
    torch.save(net.state_dict(), ckpt); return net


@torch.no_grad()
def teacher_posterior_on_current(teacher, sq, i, poses_seq, dev):
    """Return current-scan points' teacher softmax posterior + recoverability gate weight per current point."""
    acc, ast, aseg, cur = accumulate(sq, i, poses_seq)
    feat, gc, inv, _ = vox(acc, ast)
    logit = teacher(to_in(feat, gc, dev))
    post = F.softmax(logit, 1)                      # (Vacc, C)
    post_pts = post[inv]                            # per accumulated-point posterior
    cur_post = post_pts[cur]                        # current-scan points
    # recoverability gate on current points
    c0 = acc[cur]
    t1 = cKDTree(c0); n1 = t1.query_ball_point(c0, 0.5, return_length=True).astype(np.float32)
    ta = cKDTree(acc); nK = ta.query_ball_point(c0, 0.5, return_length=True).astype(np.float32)
    rec = ((n1 < 5) & (nK / np.maximum(n1, 1) >= 4)).astype(np.float32)
    conf = cur_post.max(1).values.cpu().numpy()
    w = rec * conf
    return c0, cur_post, w


def train_student(distill, iters, dev, poses_all, teacher, seed, lam=1.0, tau=2.0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_model(SP).to(dev); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.005)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.05)
    pool = [(sq, int(os.path.basename(f)[:-4])) for sq in TRAIN_SEQS for f in glob.glob(f"{KROOT}/{sq}/velodyne/*.bin")]
    t0 = time.time()
    for it in range(iters):
        sq, i = pool[np.random.randint(len(pool))]
        try:
            coord, st, seg = load(sq, i)
        except Exception:
            continue
        # student voxelization WITHOUT aug (so coords align with teacher posterior on raw current points)
        feat, gc, inv, idx = vox(coord, st, aug=False)
        logit = net(to_in(feat, gc, dev))
        segv = torch.from_numpy(seg[idx].astype(np.int64)).to(dev)
        loss = F.cross_entropy(logit, segv, ignore_index=IGNORE)
        if distill:
            try:
                c0, cur_post, w = teacher_posterior_on_current(teacher, sq, i, poses_all[sq], dev)
                # map teacher per-current-point posterior to student voxels via idx (idx indexes current points)
                tpost_v = cur_post[idx]                      # (V, C) teacher posterior at each student voxel's first point
                wv = torch.from_numpy(w[idx]).to(dev)        # (V,) gate
                s_logp = F.log_softmax(logit / tau, 1)
                kl = F.kl_div(s_logp, (tpost_v + 1e-8).pow(1 / tau) / (tpost_v + 1e-8).pow(1 / tau).sum(1, keepdim=True),
                              reduction="none").sum(1)
                if wv.sum() > 0:
                    loss = loss + lam * (wv * kl).sum() / (wv.sum() + 1e-6)
            except Exception as e:
                pass
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (it + 1) % 4000 == 0:
            print(f"  [distill={distill} seed{seed}] it{it+1} loss={loss.item():.3f} {(time.time()-t0)/(it+1)*1000:.0f}ms/it", flush=True)
    net.eval()
    return net


@torch.no_grad()
def eval_student(net, dev, poses_all, limit=150):
    cm_all, cm_rs = CM(), CM(); poses_v = poses_all[VAL_SEQ]
    files = sorted(glob.glob(f"{KROOT}/{VAL_SEQ}/velodyne/*.bin"))
    for k in range(0, min(limit * 4, len(files)), 4):
        i = k; coord, st, seg = load(VAL_SEQ, i)
        rsm = rs_mask(coord, seg, poses_v, VAL_SEQ, i)
        feat, gc, inv, _ = vox(coord, st)
        pred = net(to_in(feat, gc, dev)).argmax(1).cpu().numpy()[inv]
        cm_all.add(pred, seg); cm_rs.add(pred[rsm], seg[rsm])
    return cm_all.miou(THING), cm_rs.miou(THING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--teacher_iters", type=int, default=20000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/distill")
    a = ap.parse_args(); dev = "cuda"
    poses_all = {sq: get_poses(sq) for sq in TRAIN_SEQS + [VAL_SEQ]}
    teacher = build_teacher(a.teacher_iters, dev, poses_all)
    res = {"baseline": {}, "distill": {}}
    for distill in [0, 1]:
        key = "distill" if distill else "baseline"
        for s in a.seeds:
            net = train_student(distill, a.iters, dev, poses_all, teacher, s, lam=a.lam)
            th, rs = eval_student(net, dev, poses_all)
            res[key][f"seed{s}"] = dict(thing_mIoU=round(th, 4), RS_thing_mIoU=round(rs, 4))
            print(f"[{key} seed{s}] thing={th:.4f} RS-thing={rs:.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/distill.json", "w"), indent=1)
    print("DONE ->", f"{a.out}/distill.json", flush=True)


if __name__ == "__main__":
    main()
