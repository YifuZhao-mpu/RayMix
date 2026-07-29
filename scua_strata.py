"""Pre-registered range-class strata re-scoring (SCUA study, OJ-ITS version).

Pre-registration (fixed BEFORE looking at any result; Codex-max spec 2026-07-19):
  ranges   0-10 / 10-20 / 20-30 / >=30 m (horizontal range of the GT point, PointClip applied first)
  groups   VRU = {people, rider, bike}; small-vertical = {traffic-sign, pole, cone-stone}
  metrics  stratum-conditional IoU (I/U restricted to GT-or-pred points in the stratum is NOT used;
           we restrict by the POINT's own range bin) + GT-conditioned recall; every seed and every
           stratum is reported; scan-level PAIRED bootstrap (shared scan resample across arms) for
           arm contrasts. Labelled "operationally relevant range-class strata", not "safety".
Inference: full-point via the config's test-order pipeline (PointClip -> GridSample(return_inverse)),
best-val checkpoints of the 9 retrained single-GPU arms (internally BN-consistent).
Machine-prints outputs/poss/scua_strata.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/project/Pointcept")

from pointcept.datasets import build_dataset  # noqa: E402
from pointcept.datasets.transform import Compose  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402

PC = "/workspace/project/Pointcept"
PO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "poss")
ARMS = {"base": ["scua_base_s0", "scua_base_s1", "scua_base_s2"],
        "polarmix": ["scua_pm_s0", "scua_pm_s1", "scua_pm_s2"],
        "raymix": ["scua_rm_s0", "scua_rm_s1", "scua_rm_s2"]}
CFG = os.path.join(PC, "configs/semantic_poss/poss_spunet_seq03.py")
NCLS = 13
BINS = [(0, 10), (10, 20), (20, 30), (30, 1e9)]
BIN_NAMES = ["0-10", "10-20", "20-30", ">=30"]
GROUPS = dict(VRU=[0, 1, 11], small_vertical=[5, 6, 9])
CLIP = (-51.2, -51.2, -4, 51.2, 51.2, 2.4)
B = 2000


def clip_mask(c):
    return ((c[:, 0] > CLIP[0]) & (c[:, 0] < CLIP[3]) & (c[:, 1] > CLIP[1])
            & (c[:, 1] < CLIP[4]) & (c[:, 2] > CLIP[2]) & (c[:, 2] < CLIP[5]))


SMOKE = os.environ.get("SCUA_SMOKE", "") == "1"


def load_model(exp_name, cfg):
    model = build_model(cfg.model)
    if SMOKE:  # random-init plumbing test, no checkpoint needed
        return model.cuda().eval(), None
    ckpt = os.path.join(PC, "exp/semantic_poss", exp_name, "model", "model_best.pth")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.cuda().eval(), state.get("epoch", None)


def main():
    cfg = Config.fromfile(CFG)
    ds = build_dataset(dict(type="SemanticPOSSDataset", split="val",
                            data_root=cfg.data_root, eval_seq=3, transform=[],
                            test_mode=False, ignore_index=-1))
    tf = Compose([
        dict(type="Copy", keys_dict={"segment": "origin_segment"}),
        dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
             feat_keys=("coord", "strength")),
    ])
    n_scans = len(ds.data_list)
    print(f"[strata] val scans: {n_scans}")

    # per (arm, seed): I/U/T arrays [scan, bin, class]
    counts = {}
    for arm, exps in ARMS.items():
        for si, exp in enumerate(exps):
            model, ep = load_model(exp, cfg)
            I = np.zeros((n_scans, len(BINS), NCLS), np.int64)
            U = np.zeros_like(I)
            T = np.zeros_like(I)
            P = np.zeros_like(I)
            with torch.no_grad():
                for k in range(n_scans):
                    raw = np.fromfile(ds.data_list[k], dtype=np.float32).reshape(-1, 4)
                    coord, strength = raw[:, :3], raw[:, 3:4]
                    lab = ds.data_list[k].replace("velodyne", "labels").replace(".bin", ".label")
                    seg = ds._lut[np.clip(np.fromfile(lab, dtype=np.int32) & 0xFFFF,
                                          0, 259)].astype(np.int32)
                    m = clip_mask(coord)
                    coord, strength, seg = coord[m], strength[m], seg[m]
                    d = tf(dict(coord=coord.copy(), strength=strength.copy(),
                                segment=seg.copy()))
                    inp = dict(coord=d["coord"].cuda(non_blocking=True),
                               grid_coord=d["grid_coord"].cuda(non_blocking=True),
                               feat=d["feat"].cuda(non_blocking=True),
                               offset=torch.tensor([d["coord"].shape[0]]).cuda())
                    logits = model(inp)["seg_logits"]
                    pred = logits.argmax(1).cpu().numpy()[d["inverse"].numpy()]
                    rng = np.sqrt((coord[:, :2] ** 2).sum(1))
                    valid = seg >= 0
                    for bi, (lo, hi) in enumerate(BINS):
                        bm = valid & (rng >= lo) & (rng < hi)
                        gs, ps = seg[bm], pred[bm]
                        for c in range(NCLS):
                            g = gs == c
                            p = ps == c
                            I[k, bi, c] = (g & p).sum()
                            U[k, bi, c] = (g | p).sum()
                            T[k, bi, c] = g.sum()
                            P[k, bi, c] = p.sum()
                    if (k + 1) % 100 == 0:
                        print(f"[{arm}/s{si}] {k+1}/{n_scans}")
            counts[(arm, si)] = dict(I=I, U=U, T=T, P=P, best_epoch=ep)
            del model
            torch.cuda.empty_cache()
            print(f"[strata] done {exp}")

    def iou(I, U):
        return np.where(U > 0, I / np.maximum(U, 1), np.nan)

    def agg(cnt, scan_idx):  # pooled per (bin, class/group) over selected scans
        I = cnt["I"][scan_idx].sum(0)
        U = cnt["U"][scan_idx].sum(0)
        T = cnt["T"][scan_idx].sum(0)
        out = dict(cls_iou=iou(I, U), cls_recall=np.where(T > 0, I / np.maximum(T, 1), np.nan))
        # micro group IoU per bin
        grp = {}
        for gname, gcls in GROUPS.items():
            gi = I[:, gcls].sum(-1)
            gu = U[:, gcls].sum(-1)
            grp[gname] = np.where(gu > 0, gi / np.maximum(gu, 1), np.nan)
        out["groups_iou"] = grp
        out["all_miou"] = np.nanmean(iou(I, U), axis=-1)
        return out

    full = np.arange(n_scans)
    rng_bs = np.random.default_rng(7)
    boot_idx = [rng_bs.integers(0, n_scans, n_scans) for _ in range(B)]

    result = dict(pre_registration=dict(bins=BIN_NAMES, groups=GROUPS,
                                        note="fixed before scoring; all seeds & strata reported"),
                  arms={}, contrasts={})
    for arm in ARMS:
        seeds = []
        for si in range(3):
            a = agg(counts[(arm, si)], full)
            seeds.append(a)
        result["arms"][arm] = dict(
            best_epochs=[counts[(arm, si)]["best_epoch"] for si in range(3)],
            cls_iou_mean=np.nanmean([s["cls_iou"] for s in seeds], 0).round(4).tolist(),
            cls_iou_std=np.nanstd([s["cls_iou"] for s in seeds], 0).round(4).tolist(),
            cls_recall_mean=np.nanmean([s["cls_recall"] for s in seeds], 0).round(4).tolist(),
            groups_iou_mean={g: np.nanmean([s["groups_iou"][g] for s in seeds], 0).round(4).tolist()
                             for g in GROUPS},
            groups_iou_std={g: np.nanstd([s["groups_iou"][g] for s in seeds], 0).round(4).tolist()
                            for g in GROUPS},
            all_miou_mean=np.nanmean([s["all_miou"] for s in seeds], 0).round(4).tolist())

    # paired (shared scan resample) bootstrap for group/stratum contrasts
    def group_stat(cnt_list, idx):  # mean over seeds of micro group IoU [bin] for one arm
        vals = {g: [] for g in GROUPS}
        for cnt in cnt_list:
            I = cnt["I"][idx].sum(0)
            U = cnt["U"][idx].sum(0)
            for g, gcls in GROUPS.items():
                gi = I[:, gcls].sum(-1)
                gu = U[:, gcls].sum(-1)
                vals[g].append(np.where(gu > 0, gi / np.maximum(gu, 1), np.nan))
        return {g: np.nanmean(v, 0) for g, v in vals.items()}

    for a1, a2 in [("raymix", "polarmix"), ("raymix", "base"), ("polarmix", "base")]:
        c1 = [counts[(a1, si)] for si in range(3)]
        c2 = [counts[(a2, si)] for si in range(3)]
        point = {g: (group_stat(c1, full)[g] - group_stat(c2, full)[g]) for g in GROUPS}
        boots = {g: [] for g in GROUPS}
        for bidx in boot_idx:
            s1 = group_stat(c1, bidx)
            s2 = group_stat(c2, bidx)
            for g in GROUPS:
                boots[g].append(s1[g] - s2[g])
        result["contrasts"][f"{a1}-{a2}"] = {
            g: dict(point=(point[g] * 100).round(2).tolist(),
                    lo=(np.nanpercentile(boots[g], 2.5, axis=0) * 100).round(2).tolist(),
                    hi=(np.nanpercentile(boots[g], 97.5, axis=0) * 100).round(2).tolist())
            for g in GROUPS}

    path = os.path.join(PO, "scua_strata.json")
    json.dump(result, open(path, "w"), indent=1, default=float)
    print("wrote", path)
    for arm in ARMS:
        print(arm, "all-mIoU by bin:", result["arms"][arm]["all_miou_mean"])
    for k, v in result["contrasts"].items():
        for g in GROUPS:
            print(f"{k} {g}: point={v[g]['point']} CI=[{v[g]['lo']},{v[g]['hi']}]")


if __name__ == "__main__":
    main()
