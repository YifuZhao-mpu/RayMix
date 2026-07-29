"""Machine-print supplementary KITTI evidence for the RayMix paper audit.
Reads ONLY raw train.logs; prints kitti_perseed.json. No hand-typed numbers.
Protocol notes (matching every existing result JSON):
  - loop_best  = max over the 50/24 per-epoch evaluator.py val mIoUs (what all tables report)
  - test_precise = the single test.py line-340 precise full-point eval of the best checkpoint
  - per-class small-class IoUs = the final test.py per-class block (same source as kitti_seeds.json)
"""
import re, json, numpy as np

R = "/root/project/Pointcept/exp/semantic_kitti"
SMALL = [1, 2, 5, 6, 7, 15, 17, 18]

def parse(exp):
    txt = open(f"{R}/{exp}/train.log").read()
    loop = [float(v) * 100 for v in re.findall(r"evaluator\.py line 187 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)]
    tp = re.findall(r"test\.py line 340 \d+\] Val result: mIoU/mAcc/allAcc ([0-9.]+)", txt)
    rows = re.findall(r"test\.py line 346 \d+\] Class_(\d+)\s*-\s*[\w\- ]+\s*Result:\s*iou/accuracy\s*([0-9.]+)/", txt)[-19:]
    pc = {int(c): float(i) * 100 for c, i in rows}
    return dict(loop_best=round(max(loop), 2), test_precise=round(float(tp[-1]) * 100, 2) if tp else None,
                n_loop_evals=len(loop), pc=pc, loop_curve=[round(v, 2) for v in loop])

arms24 = {
    "baseline": ["kitti_base", "kbaseline_s1", "kbaseline_s2"],
    "balanced_naive": ["kitti_polarmix", "kbalnaive_s1", "kbalnaive_s2"],
    "raymix_full": ["kitti_raymix", "kraymix_s1", "kraymix_s2"],
}
out = {"protocol": {"loop_best": "max per-epoch evaluator val mIoU (reported in all tables)",
                    "test_precise": "test.py precise full-point eval of best checkpoint",
                    "per_class": "from the final test.py per-class block"},
       "arms_24ep": {}}

P = {}
for arm, exps in arms24.items():
    seeds = [parse(e) for e in exps]
    P[arm] = seeds
    out["arms_24ep"][arm] = {
        "exps": exps,
        "loop_best_per_seed": [s["loop_best"] for s in seeds],
        "loop_best_mean": round(float(np.mean([s["loop_best"] for s in seeds])), 4),
        "test_precise_per_seed": [s["test_precise"] for s in seeds],
        "test_precise_mean": round(float(np.mean([s["test_precise"] for s in seeds])), 4),
    }

paired = [round(r["loop_best"] - b["loop_best"], 2) for r, b in zip(P["raymix_full"], P["balanced_naive"])]
out["paired_delta_per_seed_loop"] = paired
out["delta_mean_loop_unrounded"] = round(float(np.mean([r["loop_best"] for r in P["raymix_full"]])
                                               - np.mean([b["loop_best"] for b in P["balanced_naive"]])), 4)
out["delta_mean_test_precise"] = round(out["arms_24ep"]["raymix_full"]["test_precise_mean"]
                                       - out["arms_24ep"]["balanced_naive"]["test_precise_mean"], 4)

# small classes: 3-seed mean of test.py per-class IoUs, full precision deltas
sm = {}
for c in SMALL:
    bn = float(np.mean([s["pc"][c] for s in P["balanced_naive"]]))
    rm = float(np.mean([s["pc"][c] for s in P["raymix_full"]]))
    sm[c] = {"balnaive": round(bn, 3), "raymix": round(rm, 3), "delta": round(rm - bn, 3)}
out["small_classes"] = sm
out["small_mean_delta"] = round(float(np.mean([v["delta"] for v in sm.values()])), 3)
out["small_n_strictly_up"] = int(sum(1 for v in sm.values() if v["delta"] > 0))
out["small_min_delta"] = round(min(v["delta"] for v in sm.values()), 3)

# component ablation, seed0, loop_best
comp = {"balanced_naive": P["balanced_naive"][0]["loop_best"], "plus_snap": parse("ksnap_s0")["loop_best"],
        "plus_validity": parse("kvalid_s0")["loop_best"], "raymix_full": P["raymix_full"][0]["loop_best"]}
out["component_ablation_seed0_loop_best"] = comp

# 50-epoch paired runs: curves and lead structure
bn50, rm50 = parse("kitti_balnaive_50ep"), parse("kitti_raymix_50ep")
cb, cr = bn50["loop_curve"], rm50["loop_curve"]
lead_eps = [i + 1 for i, (b, r) in enumerate(zip(cb, cr)) if r > b]
def runbest(c):
    m, o = -1, []
    for v in c:
        m = max(m, v); o.append(m)
    return o
rbb, rbr = runbest(cb), runbest(cr)
cross = next(i + 1 for i in range(len(cb)) if all(rbb[j] >= rbr[j] for j in range(i, len(cb))))
out["k50ep"] = {
    "n_epochs": len(cb),
    "balnaive": {"loop_best": bn50["loop_best"], "test_precise": bn50["test_precise"]},
    "raymix":   {"loop_best": rm50["loop_best"], "test_precise": rm50["test_precise"]},
    "delta_loop": round(rm50["loop_best"] - bn50["loop_best"], 2),
    "delta_test_precise": round(rm50["test_precise"] - bn50["test_precise"], 2),
    "epochs_raymix_leads_per_epoch": lead_eps,
    "n_lead": len(lead_eps),
    "best_so_far_balnaive_permanently_ahead_from_epoch": cross,
    "balnaive_curve": cb, "raymix_curve": cr,
}

with open("/root/Fresh_ARIS8/code/outputs/poss/kitti_perseed.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("k50ep",)}, indent=1)[:2000])
k = dict(out["k50ep"]); k.pop("balnaive_curve"); k.pop("raymix_curve")
print(json.dumps(k, indent=1))
