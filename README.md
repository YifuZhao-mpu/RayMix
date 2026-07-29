# RayMix — Sensor-Consistent Instance Mixing for LiDAR Semantic Segmentation

Code and machine-printed results for the manuscript

> **Sensor-Consistent Instance Mixing for LiDAR Semantic Segmentation in Campus Automated
> Driving: A Protocol-, Exposure-, and Convergence-Controlled Study on SemanticPOSS**
> *(under review, 2026)*

Instance-mixing augmentation defines the reported state of the art on SemanticPOSS, but its
gains entangle three things: the validation protocol, the class exposure the augmentation adds,
and its geometric fidelity to the sensor. This repository contains the code that disentangles
them — the **RayMix** augmentation itself, the **SCUA** (Sensor-Consistency–Utility Audit)
measurement harness, the controlled training scripts, and the pre-registered range–class strata
re-scoring.

Headline result the code reproduces: at matched exposure and dose, ray-consistency repairs the
measured acquisition inconsistencies of naive pasting by an order of magnitude *without* changing
converged mIoU — but it changes **where** the errors live. Naive mixing significantly degrades
far-field (≥20 m) vulnerable-road-user IoU (−2.11 and −1.96 pp, scan-bootstrap CIs excluding
zero); RayMix holds it.

---

## Repository layout

```
pointcept_patch/     the method itself — RayMix/PolarMix datasets + configs for Pointcept
scua_*.py            SCUA audit: residual metrics, per-sensor calibration, strata re-scoring
phys_*.py            physical-validity diagnostics (double returns, ground clearance, ...)
run_*.sh             training / evaluation campaigns (multi-GPU, multi-seed)
gen_*.py             paper assets generated from audited JSONs — no hand-typed numbers
paper_*_claim_check.py   machine cross-check of every number in the manuscript
outputs/**/*.json    machine-printed results: the numeric record behind every claim
v100/                a self-contained setup + run recipe for a V100 machine
```

**Start here:** `pointcept_patch/README.md` (the augmentation), then `scua_metrics.py`
(the measurement definitions), then `outputs/poss/*.json` (the results).

## The method

`pointcept_patch/pointcept/datasets/semantic_poss_raymix.py` is RayMix: an instance bank with
round-robin balanced sampling, **ground snapping**, **validity rejection**, and a **sensor-ray
z-buffer** so transplanted instances occlude and are occluded exactly as a real return from that
sensor would be. Each component is an independent config switch, which is what makes the
component ablations possible.

## The audit (SCUA)

`scua_metrics.py` defines three acquisition-consistency residuals; `scua_poss.py` and
`scua_kitti.py` calibrate them on two sensors (Pandora 40-beam, Velodyne HDL-64E) with controlled
corruptions that must move one residual and leave the others flat — a monotonicity/specificity
test the harness evaluates and prints itself. Both are CPU-only and require no training.

```bash
python3 scua_poss.py     # -> outputs/poss/scua_poss.json
python3 scua_kitti.py    # -> outputs/poss/scua_kitti.json
python3 scua_strata.py   # -> outputs/poss/scua_strata.json  (range-class strata re-scoring)
```

## Reproducing the training results

1. Clone [Pointcept](https://github.com/Pointcept/Pointcept) and apply `pointcept_patch/`
   (instructions in `pointcept_patch/README.md`).
2. Build the sparse-convolution backend — `build_torchsparse.sh` is the recipe used here
   (torch 2.7.1 + cu128).
3. Fetch [SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html) and, for the cross-sensor
   arm, [SemanticKITTI](http://semantic-kitti.org/), and link them into Pointcept's `data/`.
4. Run a campaign, e.g. the 3-arm × 3-seed strata retrain:

```bash
bash run_scua_retrain.sh          # baseline / PolarMix / RayMix, 3 seeds, single-GPU
bash run_ablations.sh             # component, decomposition and paste-volume ablations
bash run_kitti_seeds.sh           # 64-beam cross-sensor arm
bash run_ptv3_pilot.sh            # PT-v3 second-backbone check
```

> The `run_*.sh` scripts carry the absolute paths and GPU assignments of the machine they were
> run on (`PC=`, `PY=`, `LOG=`, `CUDA_VISIBLE_DEVICES`). Edit those four lines at the top before
> running them anywhere else.

## Results and the claim check

Every number in the manuscript is machine-printed into `outputs/**/*.json` and cross-checked
against the LaTeX source, so no figure in the paper is hand-transcribed:

```bash
python3 paper_tiv_v2_claim_check.py   # exit 0 = every claim matches the JSONs
```

The claim-check and `gen_*` scripts expect the manuscript sources in a sibling directory
(`../paper_tiv_v2/`), which is not part of this repository — they are included as the audit
trail for the results, not as a standalone entry point. The JSONs they read *are* here.

## Not part of the paper

`amodal_supervision.py`, `precompute_amodal.py`, `train_amodal_pilot.py`, `headroom_audit.py`,
`teacher_upperbound.py`, `converged_upperbound.py`, `distill.py`, `killshot_kd.py`,
`run_phase1_chain.sh` and `radar_audit.py` are an earlier exploratory line (amodal multi-sweep
supervision and its distillation upper-bound tests, plus a radar-support audit) that was measured,
found not to pay off, and dropped. They are kept for the record and are not needed to reproduce
anything in the manuscript.

## What is not in this repository

Training logs (~590 MB), the amodal target cache (~1.7 GB of `.npz`), and model checkpoints are
excluded by `.gitignore`. The JSON results distilled from them are tracked.

## Citation

Citation details will be added once the manuscript is accepted.

## License

MIT — see [`LICENSE`](LICENSE).

The files under `pointcept_patch/` are written to run inside
[Pointcept](https://github.com/Pointcept/Pointcept) (MIT, © 2023 Pointcept): the configs follow
Pointcept's config conventions, and `semantic_kitti_robo3d_label.diff` is a one-line patch
against upstream source, which stays under its own license. No upstream file is redistributed
here.

## Acknowledgements

Built on [Pointcept](https://github.com/Pointcept/Pointcept) (MIT). SemanticPOSS and
SemanticKITTI are distributed by their respective authors under their own terms.
