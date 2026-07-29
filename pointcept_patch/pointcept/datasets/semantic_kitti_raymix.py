"""SemanticKITTI + Ray-Consistent Instance Transplantation (RayMix) — cross-sensor (64-beam) arm for the T-ITS paper.
Self-contained (does NOT import the SemanticPOSS RayMix dataset, so editing this never affects running POSS jobs);
reuses only the two pure geometry helpers. Same mechanism as POSS RayMix: scene azimuth-swap + ray-consistent
instance transplant (ground-support + validity-reject + ray z-buffer occlusion), with KITTI 19-class ids and
HDL-64E sensor params (64 beams, elevation ~[-25, 3] deg). Purpose: show the PolarMix artifact / RayMix fix and the
thin-small rescue are NOT dataset-specific (Codex cross-sensor must-have)."""
import os
import numpy as np
from sklearn.cluster import DBSCAN

from .builder import DATASETS
from .semantic_kitti import SemanticKITTIDataset
from .semantic_poss_raymix import _polar, _rotate_z   # pure functions, safe to import

# KITTI train-ids: 0 car,1 bicycle,2 motorcycle,3 truck,4 other-veh,5 person,6 bicyclist,7 motorcyclist,
#                  8 road,9 parking,10 sidewalk,11 other-ground,12 building,13 fence,14 vegetation,
#                  15 trunk,16 terrain,17 pole,18 traffic-sign
K_TRANSPLANT = (0, 1, 2, 3, 4, 5, 6, 7, 15, 17, 18)   # movable + thin/small things
K_GROUND_SUPPORTED = {0, 1, 2, 3, 4, 5, 6, 7, 15, 17}  # base sits on ground (sign 18 may be elevated -> no snap)
K_GROUND_CLASSES = (8, 9, 10, 11, 16)                  # road, parking, sidewalk, other-ground, terrain
K_STRUCTURE = (0, 12)                                  # car, building -> cannot spawn inside


@DATASETS.register_module()
class SemanticKITTIRayMixDataset(SemanticKITTIDataset):
    def __init__(self, raymix_p=0.5, swap_p=1.0, omega_range=(np.pi / 4, np.pi),
                 transplant_classes=K_TRANSPLANT, max_instances=40,
                 n_beam=64, n_azimuth=2048, elev_range=(-25.0, 3.0),
                 dbscan_eps=0.5, dbscan_min=10, min_inst_pts=20,
                 ground_radius=2.0, snap_ground=True, validity=True, struct_reject=8,
                 use_ray_merge=True, volume_mult=1, do_transplant=True, **kwargs):
        self.raymix_p = raymix_p; self.swap_p = swap_p; self.omega_range = omega_range
        self.transplant_classes = list(transplant_classes); self.max_instances = max_instances
        self.n_beam = n_beam; self.n_azimuth = n_azimuth
        self.elev_lo = np.radians(elev_range[0]); self.elev_hi = np.radians(elev_range[1])
        self.dbscan_eps = dbscan_eps; self.dbscan_min = dbscan_min; self.min_inst_pts = min_inst_pts
        self.ground_radius = ground_radius; self.snap_ground = snap_ground
        self.validity = validity; self.struct_reject = struct_reject
        self.use_ray_merge = use_ray_merge; self.volume_mult = volume_mult; self.do_transplant = do_transplant
        super().__init__(**kwargs)

    def _cell(self, coord):
        r, theta, phi = _polar(coord)
        beam = np.clip(((phi - self.elev_lo) / (self.elev_hi - self.elev_lo) * self.n_beam).astype(np.int64),
                       0, self.n_beam - 1)
        az = np.clip(((theta + np.pi) / (2 * np.pi) * self.n_azimuth).astype(np.int64), 0, self.n_azimuth - 1)
        return beam * self.n_azimuth + az, r

    def _load_raw(self, path):
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3].astype(np.float32); strength = scan[:, 3:4].astype(np.float32)
        lab = path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(lab):
            raw = np.fromfile(lab, dtype=np.int32).reshape(-1) & 0xFFFF
            seg = np.vectorize(self.learning_map.__getitem__)(raw).astype(np.int32)
        else:
            seg = np.zeros(len(coord), np.int32)
        return coord, strength, seg

    def _is_ground(self, seg):
        return np.isin(seg, K_GROUND_CLASSES)

    def _swap_sector(self, cA, sA, tA, cB, sB, tB):
        w = np.random.uniform(*self.omega_range); alpha = np.random.uniform(-np.pi, np.pi - w); beta = alpha + w
        _, yawA, _ = _polar(cA); _, yawB, _ = _polar(cB)
        inA = (yawA > alpha) & (yawA < beta); inB = (yawB > alpha) & (yawB < beta)
        return (np.concatenate([cA[~inA], cB[inB]], 0), np.concatenate([sA[~inA], sB[inB]], 0),
                np.concatenate([tA[~inA], tB[inB]], 0))

    def _extract_instances(self, cB, sB, tB):
        gmask = self._is_ground(tB); gz = cB[gmask] if gmask.any() else None
        per_class = {}
        for c in self.transplant_classes:
            m = tB == c
            if m.sum() < self.min_inst_pts:
                continue
            pts, st = cB[m], sB[m]
            labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min).fit(pts).labels_
            bucket = []
            for k in np.unique(labels):
                if k == -1:
                    continue
                sel = labels == k
                if sel.sum() < self.min_inst_pts:
                    continue
                ip, ist = pts[sel], st[sel]; base_z = float(ip[:, 2].min()); clear = 0.0
                if gz is not None:
                    cen = ip[:, :2].mean(0)
                    near = gz[np.sum((gz[:, :2] - cen) ** 2, 1) < self.ground_radius ** 2]
                    if len(near):
                        clear = base_z - float(np.median(near[:, 2]))
                bucket.append((ip.copy(), ist.copy(), int(c), base_z, clear))
            if bucket:
                np.random.shuffle(bucket); per_class[c] = bucket
        insts, classes, i = [], list(per_class.keys()), 0
        while len(insts) < self.max_instances and classes:
            c = classes[i % len(classes)]
            if per_class[c]:
                insts.append(per_class[c].pop())
            else:
                classes.remove(c); continue
            i += 1
        return insts

    def _place(self, inst, cA, tA):
        ip, ist, c, base_z, clear = inst
        theta = np.random.uniform(-np.pi, np.pi)
        rp = _rotate_z(ip, theta).astype(np.float32); new_cen = rp[:, :2].mean(0)
        if self.snap_ground and c in K_GROUND_SUPPORTED:
            gmask = self._is_ground(tA)
            if not gmask.any():
                return None
            gz = cA[gmask]; near = gz[np.sum((gz[:, :2] - new_cen) ** 2, 1) < self.ground_radius ** 2]
            if not len(near):
                return None
            rp[:, 2] += (float(np.median(near[:, 2])) + clear) - float(rp[:, 2].min())
        if self.validity:
            sm = np.isin(tA, K_STRUCTURE)
            if sm.any():
                hs = cA[sm]; zmin, zmax = rp[:, 2].min() - 0.2, rp[:, 2].max() + 0.2
                box = hs[(hs[:, 2] > zmin) & (hs[:, 2] < zmax)]
                if len(box) and (np.sum((box[:, :2] - new_cen) ** 2, 1) < 0.5 ** 2).sum() >= self.struct_reject:
                    return None
        return rp.astype(np.float32), ist.astype(np.float32), c

    def _ray_merge(self, cA, sA, tA, add_c, add_s, add_t):
        cellA, rA = self._cell(cA); cellI, rI = self._cell(add_c)
        mx = self.n_beam * self.n_azimuth
        host_min = np.full(mx, np.inf, np.float32); np.minimum.at(host_min, cellA, rA)
        inst_min = np.full(mx, np.inf, np.float32); np.minimum.at(inst_min, cellI, rI)
        inst_keep = rI < host_min[cellI]; host_del = rA > inst_min[cellA]
        return (np.concatenate([cA[~host_del], add_c[inst_keep]], 0).astype(np.float32),
                np.concatenate([sA[~host_del], add_s[inst_keep]], 0).astype(np.float32),
                np.concatenate([tA[~host_del], add_t[inst_keep]], 0).astype(np.int32))

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        cA, sA, tA = self._load_raw(path)
        if np.random.rand() < self.raymix_p:
            cB, sB, tB = self._load_raw(self.data_list[np.random.randint(len(self.data_list))])
            if np.random.rand() < self.swap_p:
                cA, sA, tA = self._swap_sector(cA, sA, tA, cB, sB, tB)
            insts = self._extract_instances(cB, sB, tB) if self.do_transplant else []
            add_c, add_s, add_t = [], [], []
            for inst in insts:
                for _ in range(self.volume_mult):
                    placed = self._place(inst, cA, tA)
                    if placed is None:
                        continue
                    pc, ps, pcl = placed
                    add_c.append(pc); add_s.append(ps); add_t.append(np.full(len(pc), pcl, np.int32))
            if add_c:
                A_c, A_s, A_t = np.concatenate(add_c, 0), np.concatenate(add_s, 0), np.concatenate(add_t, 0)
                if self.use_ray_merge:
                    cA, sA, tA = self._ray_merge(cA, sA, tA, A_c, A_s, A_t)
                else:
                    cA = np.concatenate([cA, A_c], 0).astype(np.float32)
                    sA = np.concatenate([sA, A_s], 0).astype(np.float32)
                    tA = np.concatenate([tA, A_t], 0).astype(np.int32)
        return dict(coord=cA.astype(np.float32), strength=sA.astype(np.float32),
                    segment=tA.astype(np.int32), name=self.get_data_name(idx))
