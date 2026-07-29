#!/usr/bin/env python3
"""Machine-print the 9 retrained single-GPU arms (best-val mIoU parsed from train logs) with
3-seed stats and paired contrasts; anchors against the published 1-GPU 50-ep PolarMix references.
Writes outputs/poss/scua_retrain.json. Never hand-type these numbers."""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PO = os.path.join(HERE, "outputs", "poss")
EXP = "/workspace/project/Pointcept/exp/semantic_poss"
T95 = 4.302653
ARMS = {"base": ["scua_base_s0", "scua_base_s1", "scua_base_s2"],
        "polarmix": ["scua_pm_s0", "scua_pm_s1", "scua_pm_s2"],
        "raymix": ["scua_rm_s0", "scua_rm_s1", "scua_rm_s2"]}


def best(exp):
    log = os.path.join(EXP, exp, "train.log")
    vals = re.findall(r"Best validation mIoU updated to: ([0-9.]+)", open(log).read())
    assert vals, exp
    return round(float(vals[-1]) * 100, 2)


def stats(v):
    v = np.asarray(v, float)
    return dict(seeds=v.tolist(), mean=round(float(v.mean()), 2),
                std=round(float(v.std()), 2))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return dict(per_seed=[round(float(x), 2) for x in d], mean=round(float(d.mean()), 2),
                ci95=[round(float(d.mean() - T95 * sem), 2),
                      round(float(d.mean() + T95 * sem), 2)])


arms = {a: [best(e) for e in exps] for a, exps in ARMS.items()}
out = dict(
    config="single-GPU, per-device batch 12, 50 ep, seeds {0,1,2}; all arms BN-consistent",
    published_1gpu_50ep_polarmix_refs=[59.98, 60.33],
    arms={a: stats(v) for a, v in arms.items()},
    contrasts=dict(polarmix_minus_base=paired(arms["polarmix"], arms["base"]),
                   raymix_minus_polarmix=paired(arms["raymix"], arms["polarmix"]),
                   raymix_minus_base=paired(arms["raymix"], arms["base"])))
json.dump(out, open(os.path.join(PO, "scua_retrain.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
