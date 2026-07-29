# Pointcept patch — RayMix / PolarMix / SemanticPOSS

The training-time method lives here. These files are dropped into a
[Pointcept](https://github.com/Pointcept/Pointcept) checkout; everything in the parent
directory (audit harnesses, run scripts, claim checks) reads the results those runs produce.

## Contents

| File | Role |
| --- | --- |
| `pointcept/datasets/semantic_poss_raymix.py` | **RayMix** — ray-consistent instance mixing: instance bank, round-robin balanced sampling, ground snapping, validity rejection, sensor-ray z-buffer |
| `pointcept/datasets/semantic_kitti_raymix.py` | RayMix for SemanticKITTI (64-beam cross-sensor arm) |
| `pointcept/datasets/semantic_poss_polarmix.py` | PolarMix instance branch, the comparison arm |
| `pointcept/datasets/semantic_poss.py` | SemanticPOSS dataset (both validation protocols: seq02 and seq03) |
| `configs/semantic_poss/*.py` | POSS configs — SpUNet / SPVCNN / PT-v3 x baseline / PolarMix / RayMix |
| `configs/semantic_kitti/*raymix*.py`, `kitti_spunet_base_fast.py` | KITTI configs for the cross-sensor arm |
| `semantic_kitti_robo3d_label.diff` | One-line upstream fix: map the Robo3D SemanticKITTI-C crosstalk marker (label 23) to `ignore_index` |

The ablation switches (`snap`, `validity`, `zbuffer`, paste dose, bank balancing) are dataset
kwargs — see the `data.train` block of `configs/semantic_poss/poss_spunet_raymix_seq03.py`.

## Applying it

```bash
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept
cp -r /path/to/this/pointcept_patch/pointcept/datasets/*.py pointcept/datasets/
cp -r /path/to/this/pointcept_patch/configs/* configs/
git apply /path/to/this/pointcept_patch/semantic_kitti_robo3d_label.diff   # optional, only for the corruption arm
```

Then register the datasets by adding these lines to `pointcept/datasets/__init__.py`, next to
the existing `from .semantic_kitti import SemanticKITTIDataset`:

```python
from .semantic_poss import SemanticPOSSDataset
from .semantic_poss_polarmix import SemanticPOSSPolarMixDataset
from .semantic_poss_raymix import SemanticPOSSRayMixDataset
from .semantic_kitti_raymix import SemanticKITTIRayMixDataset
```

Pointcept itself is MIT-licensed; only the files listed above are ours, and the `.diff` is the
sole modification to upstream code required by the paper's experiments.
