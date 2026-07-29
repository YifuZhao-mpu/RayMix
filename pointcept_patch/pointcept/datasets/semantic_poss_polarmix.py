"""SemanticPOSS + PolarMix augmentation (Fresh_ARIS8 L2 baseline-strength control).
Faithful PolarMix (Xiao et al., NeurIPS 2022): given scan A (current) and a random scan B from the SAME split,
(1) SCENE-LEVEL azimuth swap: cut a random azimuth sector [alpha,beta] from B and paste into A (and vice-versa,
    we keep A's complement + B's sector);
(2) INSTANCE-LEVEL rotate-paste: take points of RARE classes from B, rotate by one of a few angles, paste into A.
Applied at the raw-point level inside get_data, BEFORE the standard transforms — so it composes with the same
recipe as the from-scratch baseline (controlled comparison). p = probability of applying PolarMix per sample.
Rare (instance) classes for SemanticPOSS = the minority/thing-like set; we use the official PolarMix choice of
'instance classes' (people, rider, car, trunk, pole, trashcan, cone-stone, bike) — train-id list below."""
import os
import numpy as np

from .builder import DATASETS
from .semantic_poss import SemanticPOSSDataset

# train-ids (0..12): people0 rider1 car2 trunk3 plants4 sign5 pole6 trashcan7 building8 cone9 fence10 bike11 ground12
# PolarMix instance (rotate-paste) classes = movable / minority objects (exclude stuff: plants/building/fence/ground/sign)
POSS_INSTANCE_CLASSES = [0, 1, 2, 3, 6, 7, 9, 11]   # people, rider, car, trunk, pole, trashcan, cone, bike


def _azimuth(points):
    return np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]


def _swap_sector(cA, sA, tA, cB, sB, tB, alpha, beta):
    """Scene-level: keep A's points OUTSIDE [alpha,beta] + B's points INSIDE [alpha,beta]."""
    yawA, yawB = _azimuth(cA), _azimuth(cB)
    inA = (yawA > alpha) & (yawA < beta)
    inB = (yawB > alpha) & (yawB < beta)
    c = np.concatenate([cA[~inA], cB[inB]], 0)
    s = np.concatenate([sA[~inA], sB[inB]], 0)
    t = np.concatenate([tA[~inA], tB[inB]], 0)
    return c, s, t


def _rotate_z(coord, theta):
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], np.float32)
    return coord @ R.T


@DATASETS.register_module()
class SemanticPOSSPolarMixDataset(SemanticPOSSDataset):
    def __init__(self, polarmix_p=0.5, instance_rotations=(np.pi / 2, np.pi, np.pi * 3 / 2),
                 omega_range=(np.pi / 4, np.pi), **kwargs):
        self.polarmix_p = polarmix_p
        self.instance_rotations = instance_rotations
        self.omega_range = omega_range
        super().__init__(**kwargs)

    def _load_raw(self, path):
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3].astype(np.float32)
        strength = scan[:, 3:4].astype(np.float32)
        lab = path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(lab):
            raw = np.fromfile(lab, dtype=np.int32).reshape(-1) & 0xFFFF
            seg = self._lut[np.clip(raw, 0, 259)].astype(np.int32)
        else:
            seg = np.zeros(len(coord), np.int32)
        return coord, strength, seg

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        cA, sA, tA = self._load_raw(path)
        # train-mode only: apply PolarMix with prob p (val/test datasets use the plain SemanticPOSSDataset)
        if np.random.rand() < self.polarmix_p:
            jpath = self.data_list[np.random.randint(len(self.data_list))]
            cB, sB, tB = self._load_raw(jpath)
            # (1) scene-level azimuth-sector swap
            w = np.random.uniform(*self.omega_range)
            alpha = np.random.uniform(-np.pi, np.pi - w)
            beta = alpha + w
            cA, sA, tA = _swap_sector(cA, sA, tA, cB, sB, tB, alpha, beta)
            # (2) instance-level rotate-paste: rare-class points from B, rotated, appended to A
            inst = np.isin(tB, POSS_INSTANCE_CLASSES)
            if inst.sum() > 0:
                ci, si, ti = cB[inst], sB[inst], tB[inst]
                add_c, add_s, add_t = [], [], []
                for theta in self.instance_rotations:
                    add_c.append(_rotate_z(ci, theta)); add_s.append(si); add_t.append(ti)
                cA = np.concatenate([cA] + add_c, 0).astype(np.float32)
                sA = np.concatenate([sA] + add_s, 0).astype(np.float32)
                tA = np.concatenate([tA] + add_t, 0).astype(np.int32)
        return dict(coord=cA, strength=sA, segment=tA, name=self.get_data_name(idx))
