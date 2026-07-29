# Running the seq02 re-run + SPVCNN repro on a 4× V100 box

Goal: reproduce **seq02 3-seed** (base/PolarMix/RayMix) and the **SPVCNN+PolarMix repro** on 4× V100,
using `--num-gpus 4` so per-GPU BN batch = 3 (matches the paper's seq03 4-GPU regime). V100 NCCL is reliable
and torchsparse builds easily for sm_70, so this box solves both the multi-GPU crash and the SPVCNN blocker.

---
## 0. What I need from you to drive this (pick one)
- **(A) SSH access** to the V100 box (host, user, key) **and** confirmation the two boxes can reach each other
  (for rsync) → I do everything from here.
- **(B) You run the commands** below on the V100 box (copy-paste) — I'll interpret outputs.
- **(C) Start a Claude Code session on the V100 box** → it does the local half; I hand it this folder.

Also tell me whether the V100 box **already has** Pointcept and/or the SemanticPOSS data (saves transfer).

---
## 1. Transfer code + data  (~4.8 GB total)
From THIS Blackwell box (`/root/project`). Replace `V100` with `user@host`.

```bash
# (a) Pointcept CODE — 778 MB; EXCLUDE exp/ (334 GB!) and data/
rsync -av --exclude=exp --exclude=data /root/project/Pointcept/  V100:~/Pointcept/
# (b) SemanticPOSS DATA — 4.0 GB (follow the symlink to the real dir)
rsync -avL /root/project/data/SemanticPOSS/  V100:~/data/SemanticPOSS/
```

If the boxes can't reach each other, use the small custom-files bundle instead (needs a fresh stock
Pointcept @ commit `ef6817b` on the V100 box), then transfer data via your shared storage:
```bash
# on V100: git clone https://github.com/Pointcept/Pointcept ~/Pointcept && cd ~/Pointcept && git checkout ef6817b
# then overlay this project's custom files:
scp /root/Fresh_ARIS8/code/v100/pointcept_custom.tar.gz V100:~/ && ssh V100 'tar -xzf ~/pointcept_custom.tar.gz -C ~/Pointcept'
```
> The bundle contains the 4 custom dataset classes, `datasets/__init__.py`, all 9 `configs/semantic_poss/*.py`.
> rsync of the whole repo is safer (catches any other local mods); use the bundle only as a fallback.

---
## 2. Python env on the V100 box
**Recommended — clone the exact env via conda-pack (no dependency-resolution pain):**
```bash
# on Blackwell box:
/root/miniconda3/bin/conda install -n base -y conda-pack
/root/miniconda3/bin/conda-pack -n pointcept -o /tmp/pointcept_env.tar.gz   # a few GB
rsync -av /tmp/pointcept_env.tar.gz V100:~/
# on V100 box:
mkdir -p ~/envs/pointcept && tar -xzf ~/pointcept_env.tar.gz -C ~/envs/pointcept
source ~/envs/pointcept/bin/activate && conda-unpack
```
**Or build fresh** (py3.9): `torch==2.7.1` (cu12x wheel — includes sm_70), `spconv-cu124==2.3.8`,
`addict h5py SharedArray timm yacs scipy scikit-learn` + Pointcept's other reqs.

Sanity: `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.device_count())"`  → 4× V100.

---
## 3. torchsparse (ONLY needed for the SPVCNN repro — seq02 does NOT need it)
```bash
sudo apt-get install -y libsparsehash-dev    # or: conda install -c bioconda google-sparsehash
TORCH_CUDA_ARCH_LIST="7.0" FORCE_CUDA=1 pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git
```
⚠ **Version caveat:** Pointcept's `SPVCNN` must match the torchsparse API it was written against. If the SPVCNN
config import fails after install, check `pointcept/models/.../spvcnn*` for the expected torchsparse version and
pin it. If torchsparse is too painful, skip SPVCNN and use the existing **MinkUNet+PolarMix 57.4** (≈ our SpUNet
architecture) as the matched-architecture comparison — seq02 already carries the main correction.

---
## 4. Sanity smoke (1–2 min) before the full run
```bash
cd ~/Pointcept && export PYTHONPATH=~/Pointcept NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --config-file configs/semantic_poss/poss_spunet_seq03.py \
  --num-gpus 4 --options seed=0 epoch=2 eval_epoch=2 \
  data.train.eval_seq=2 data.val.eval_seq=2 data.test.eval_seq=2 \
  data.train.data_root=~/data/SemanticPOSS data.val.data_root=~/data/SemanticPOSS data.test.data_root=~/data/SemanticPOSS \
  save_path=exp/semantic_poss/smoke
# expect "Train: [1/2][..]" with falling loss and no "illegal memory access"
```

---
## 5. Full run
`run_all_v100.sh` (in this folder) runs seq02 (9 runs) + SPVCNN repro (3 runs) at `--num-gpus 4`, then writes
`seq02_3seed.json` and `repro_spvcnn_seq03.json`. ~2.5 h total.
```bash
scp /root/Fresh_ARIS8/code/v100/run_all_v100.sh V100:~/Pointcept/
ssh V100 'cd ~/Pointcept && PC=~/Pointcept DATA=~/data/SemanticPOSS PY=$(which python) GPUS=0,1,2,3 OUT=~/q2_results bash run_all_v100.sh'
```

---
## 6. Bring results back
```bash
rsync -av V100:~/q2_results/{seq02_3seed.json,repro_spvcnn_seq03.json}  /root/Fresh_ARIS8/code/outputs/poss/
```
Then I'll: relabel Table 2 (seq02 = recipe/no-mixing headline + PolarMix-hurts row), update abstract/intro/limitations
with the 4-GPU 3-seed numbers, drop the single-GPU caveat, and fold in the SPVCNN matched-condition result.

---
## 7. Why this fixes the open issues
- **4 GPUs** → batch 12, 3/GPU, `sync_bn=False` → BN regime identical to the paper's seq03 (kills the single-GPU confound).
- **V100 NCCL** reliable → real multi-GPU (no `illegal memory access`).
- **torchsparse on sm_70** → SPVCNN repro runs (matched-condition "beat the same method").
- Residual caveat: V100 ≠ Blackwell HW used for seq03; the dominant factor (4-GPU BN/batch regime) matches, note it.
