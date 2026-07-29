"""SemanticPOSS + Ray-Consistent Instance Transplantation (Fresh_ARIS8 L3, the proposed mechanism).

Motivation (machine-measured on our own SpUNet+PolarMix run, seq03):
  PolarMix's instance rotate-paste massively helps large movable objects (car +20.4 IoU) but barely helps —
  and even HURTS — thin/small structures: pole +3.3 (still 36.4), traffic-sign +2.2 (47.7), cone-stone -5.6
  (regressed to 44.5), trashcan 37.7. Cause (confirmed by review): naive rotate-paste copies an object's
  captured point set and drops it at a new azimuth WITHOUT respecting LiDAR ray geometry — producing
  physically impossible instances: double returns along a ray, instances floating off the ground, and
  instances spawned inside walls/cars. For thin/small classes this injects label noise instead of signal.

Proposed fix — re-render each transplanted instance through the host sensor's ray model:
  (1) keep PolarMix's SCENE-level azimuth-sector swap unchanged (this is what helps car etc.);
  (2) replace the instance paste with RAY-CONSISTENT transplantation of thin/small rare classes:
      a. extract real instances from a donor scan (DBSCAN per class);
      b. rotate around the sensor z-axis (preserves range & radial point density) to a new azimuth;
      c. SUPPORT: snap the instance base onto the host local ground (median z of host 'ground' points
         under the new footprint), preserving the instance's original ground clearance; reject if no
         ground support is found for ground-supported classes;
      d. VALIDITY: reject instances that would spawn inside dense host structure (building/car) or that
         are too sparse to be informative;
      e. OCCLUSION (ray z-buffer): assign every point to a (beam, azimuth) ray cell; an instance point
         survives only if it is the NEAREST return along its ray (drops instance points occluded by a
         closer host return), and host points strictly behind an accepted instance point are deleted
         (the instance now occludes them). This enforces one-return-per-ray physical consistency.

This is an ablation-clean drop-in: SpUNet + {no-aug, PolarMix, RayMix} share the same backbone/recipe, so
the delta isolates the instance mechanism (PolarMix naive paste  ->  ray-consistent transplant)."""
import os
import numpy as np
from sklearn.cluster import DBSCAN

from .builder import DATASETS
from .semantic_poss import SemanticPOSSDataset

# train-ids: people0 rider1 car2 trunk3 plants4 sign5 pole6 trashcan7 building8 cone9 fence10 bike11 ground12
GROUND_SUPPORTED = {1, 3, 5, 6, 7, 9, 11}   # rider,trunk,sign(on pole/ground),pole,trashcan,cone,bike base ~ ground
STRUCTURE_CLASSES = (2, 8)                   # car, building -> cannot spawn an instance inside these


def _polar(coord):
    """Return (range r, azimuth theta in [-pi,pi], elevation phi)."""
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    r = np.sqrt(rho ** 2 + coord[:, 2] ** 2) + 1e-6
    theta = np.arctan2(coord[:, 1], coord[:, 0])
    phi = np.arctan2(coord[:, 2], rho + 1e-6)
    return r, theta, phi


def _rotate_z(coord, theta):
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], np.float32)
    return coord @ R.T


@DATASETS.register_module()
class SemanticPOSSRayMixDataset(SemanticPOSSDataset):
    def __init__(self, raymix_p=0.5, swap_p=1.0, omega_range=(np.pi / 4, np.pi),
                 transplant_classes=(0, 1, 2, 3, 6, 7, 9, 11), max_instances=30,
                 n_beam=40, n_azimuth=1800, elev_range=(-16.0, 8.0),
                 dbscan_eps=0.5, dbscan_min=10, min_inst_pts=20,
                 ground_class=12, ground_radius=2.0, snap_ground=True,
                 validity=True, struct_reject=8, use_ray_merge=True,
                 volume_mult=1, do_transplant=True, **kwargs):
        self.raymix_p = raymix_p
        self.swap_p = swap_p
        self.omega_range = omega_range
        self.transplant_classes = list(transplant_classes)
        self.max_instances = max_instances
        self.n_beam = n_beam
        self.n_azimuth = n_azimuth
        self.elev_lo = np.radians(elev_range[0])
        self.elev_hi = np.radians(elev_range[1])
        self.dbscan_eps = dbscan_eps
        self.dbscan_min = dbscan_min
        self.min_inst_pts = min_inst_pts
        self.ground_class = ground_class
        self.ground_radius = ground_radius
        self.snap_ground = snap_ground
        self.validity = validity
        self.struct_reject = struct_reject
        self.use_ray_merge = use_ray_merge   # ablation: off = naive concat (no occlusion z-buffer)
        self.volume_mult = volume_mult       # volume-control: paste each accepted instance at this many random yaws
        self.do_transplant = do_transplant   # decomposition: False = scene-swap-only arm
        super().__init__(**kwargs)

    # ---- sensor ray model -------------------------------------------------
    def _cell(self, coord):
        r, theta, phi = _polar(coord)
        beam = np.clip(((phi - self.elev_lo) / (self.elev_hi - self.elev_lo) * self.n_beam).astype(np.int64),
                       0, self.n_beam - 1)
        az = np.clip(((theta + np.pi) / (2 * np.pi) * self.n_azimuth).astype(np.int64),
                     0, self.n_azimuth - 1)
        return beam * self.n_azimuth + az, r

    # ---- donor I/O --------------------------------------------------------
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

    def _swap_sector(self, cA, sA, tA, cB, sB, tB):
        w = np.random.uniform(*self.omega_range)
        alpha = np.random.uniform(-np.pi, np.pi - w)
        beta = alpha + w
        _, yawA, _ = _polar(cA)
        _, yawB, _ = _polar(cB)
        inA = (yawA > alpha) & (yawA < beta)
        inB = (yawB > alpha) & (yawB < beta)
        return (np.concatenate([cA[~inA], cB[inB]], 0),
                np.concatenate([sA[~inA], sB[inB]], 0),
                np.concatenate([tA[~inA], tB[inB]], 0))

    def _extract_instances(self, cB, sB, tB):
        """DBSCAN donor rare-class points into instances; record base z & ground clearance."""
        gmask = tB == self.ground_class
        gz = cB[gmask] if gmask.any() else None
        per_class = {}                                   # class -> list of instances
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
                ip, ist = pts[sel], st[sel]
                base_z = float(ip[:, 2].min())
                clear = 0.0                               # original ground clearance
                if gz is not None:
                    cen = ip[:, :2].mean(0)
                    near = gz[np.sum((gz[:, :2] - cen) ** 2, 1) < self.ground_radius ** 2]
                    if len(near):
                        clear = base_z - float(np.median(near[:, 2]))
                bucket.append((ip.copy(), ist.copy(), int(c), base_z, clear))
            if bucket:
                np.random.shuffle(bucket)
                per_class[c] = bucket
        # round-robin across classes so rare classes (cone/trashcan) are never crowded out by common ones
        insts, classes = [], list(per_class.keys())
        i = 0
        while len(insts) < self.max_instances and classes:
            c = classes[i % len(classes)]
            if per_class[c]:
                insts.append(per_class[c].pop())
            else:
                classes.remove(c)
                continue
            i += 1
        return insts

    def _place(self, inst, cA, tA):
        """Rotate by random yaw, snap to host ground, validity-check. Return placed (coord,strength,label) or None."""
        ip, ist, c, base_z, clear = inst
        # rotate the instance about the SENSOR z-axis: preserves each point's range & the instance's
        # internal radial point density, only moves it to a new azimuth (Codex's prescription).
        theta = np.random.uniform(-np.pi, np.pi)
        rp = _rotate_z(ip, theta).astype(np.float32)
        new_cen = rp[:, :2].mean(0)
        # SUPPORT: snap base onto host local ground
        if self.snap_ground and c in GROUND_SUPPORTED:
            gmask = tA == self.ground_class
            if not gmask.any():
                return None
            gz = cA[gmask]
            near = gz[np.sum((gz[:, :2] - new_cen) ** 2, 1) < self.ground_radius ** 2]
            if not len(near):
                return None                                        # no ground support -> reject
            target_base = float(np.median(near[:, 2])) + clear
            rp[:, 2] += target_base - float(rp[:, 2].min())
        # VALIDITY: do not spawn inside dense host structure (building/car)
        if self.validity:
            sm = np.isin(tA, STRUCTURE_CLASSES)
            if sm.any():
                hs = cA[sm]
                zmin, zmax = rp[:, 2].min() - 0.2, rp[:, 2].max() + 0.2
                box = hs[(hs[:, 2] > zmin) & (hs[:, 2] < zmax)]
                if len(box):
                    d2 = np.sum((box[:, :2] - new_cen) ** 2, 1)
                    if (d2 < 0.5 ** 2).sum() >= self.struct_reject:
                        return None                                # inside a wall/car -> reject
        return rp.astype(np.float32), ist.astype(np.float32), c

    def _ray_merge(self, cA, sA, tA, add_c, add_s, add_t):
        """Ray z-buffer: keep instance pts that are nearest along their ray; delete host pts occluded behind them."""
        cellA, rA = self._cell(cA)
        cellI, rI = self._cell(add_c)
        maxcell = self.n_beam * self.n_azimuth
        host_min = np.full(maxcell, np.inf, np.float32)
        np.minimum.at(host_min, cellA, rA)
        inst_min = np.full(maxcell, np.inf, np.float32)
        np.minimum.at(inst_min, cellI, rI)
        inst_keep = rI < host_min[cellI]            # instance survives only if nearest return on its ray
        host_del = rA > inst_min[cellA]             # host pt behind an accepted instance -> deleted
        cA2, sA2, tA2 = cA[~host_del], sA[~host_del], tA[~host_del]
        return (np.concatenate([cA2, add_c[inst_keep]], 0).astype(np.float32),
                np.concatenate([sA2, add_s[inst_keep]], 0).astype(np.float32),
                np.concatenate([tA2, add_t[inst_keep]], 0).astype(np.int32))

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        cA, sA, tA = self._load_raw(path)
        if np.random.rand() < self.raymix_p:
            jpath = self.data_list[np.random.randint(len(self.data_list))]
            cB, sB, tB = self._load_raw(jpath)
            if np.random.rand() < self.swap_p:
                cA, sA, tA = self._swap_sector(cA, sA, tA, cB, sB, tB)
            insts = self._extract_instances(cB, sB, tB) if self.do_transplant else []
            add_c, add_s, add_t = [], [], []
            for inst in insts:
                for _ in range(self.volume_mult):           # volume-control: N yaws per accepted instance
                    placed = self._place(inst, cA, tA)
                    if placed is None:
                        continue
                    pc, ps, pcl = placed
                    add_c.append(pc)
                    add_s.append(ps)
                    add_t.append(np.full(len(pc), pcl, np.int32))
            if add_c:
                A_c, A_s, A_t = np.concatenate(add_c, 0), np.concatenate(add_s, 0), np.concatenate(add_t, 0)
                if self.use_ray_merge:
                    cA, sA, tA = self._ray_merge(cA, sA, tA, A_c, A_s, A_t)   # occlusion-consistent
                else:                                                         # ablation: naive concat
                    cA = np.concatenate([cA, A_c], 0).astype(np.float32)
                    sA = np.concatenate([sA, A_s], 0).astype(np.float32)
                    tA = np.concatenate([tA, A_t], 0).astype(np.int32)
        return dict(coord=cA.astype(np.float32), strength=sA.astype(np.float32),
                    segment=tA.astype(np.int32), name=self.get_data_name(idx))
