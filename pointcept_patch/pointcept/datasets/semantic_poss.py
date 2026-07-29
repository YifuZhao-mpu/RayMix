"""SemanticPOSS dataset (Fresh_ARIS8 beat-SOTA study). Official 13-class protocol (people/rider/car/trunk/
plants/traffic-sign/pole/trashcan/building/cone-stone/fence/bike/ground); train = seq 00-05 except 03, val = 03.
Map verified against the canonical semantic-poss.yaml. KITTI-like .bin (x,y,z,intensity) + .label."""
import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset

POSS_LEARNING_MAP = {
    0: -1, 1: -1, 2: -1, 3: -1,
    4: 0, 5: 0,        # people
    6: 1,              # rider
    7: 2,              # car
    8: 3,              # trunk
    9: 4,              # plants
    10: 5, 11: 5, 12: 5,  # traffic-sign
    13: 6,             # pole
    14: 7,             # trashcan
    15: 8,             # building
    16: 9,             # cone-stone
    17: 10,            # fence
    18: -1, 19: -1, 20: -1,
    21: 11,            # bike
    22: 12,            # ground
}
POSS_NAMES = ["people", "rider", "car", "trunk", "plants", "traffic-sign", "pole",
              "trashcan", "building", "cone-stone", "fence", "bike", "ground"]


@DATASETS.register_module()
class SemanticPOSSDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, eval_seq=2, **kwargs):
        self.ignore_index = ignore_index
        self.eval_seq = eval_seq   # 2 (FRNet protocol) or 3 (PolarMix/SemanticPOSS-orig protocol)
        self.learning_map = {k: (ignore_index if v == -1 else v) for k, v in POSS_LEARNING_MAP.items()}
        # full LUT for fast vectorized mapping (raw 0..255 -> train id / ignore)
        self._lut = np.full(260, ignore_index, dtype=np.int64)
        for k, v in self.learning_map.items():
            self._lut[k] = v
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        # Two literature protocols: FRNet uses seq02; PolarMix/SemanticPOSS-orig use seq03. eval_seq selects.
        e = self.eval_seq
        train_seqs = [s for s in [0, 1, 2, 3, 4, 5] if s != e]
        split2seq = dict(train=train_seqs, val=[e], test=[e])
        seqs = []
        if isinstance(self.split, str):
            seqs = split2seq[self.split]
        else:
            for sp in self.split:
                seqs += split2seq[sp]
        data_list = []
        for seq in seqs:
            seq = str(seq).zfill(2)
            folder = os.path.join(self.data_root, "sequences", seq, "velodyne")
            for f in sorted(os.listdir(folder)):
                data_list.append(os.path.join(folder, f))
        return data_list

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])
        label_file = path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            raw = np.fromfile(label_file, dtype=np.int32).reshape(-1) & 0xFFFF
            segment = self._lut[np.clip(raw, 0, 259)].astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0], dtype=np.int32)
        return dict(coord=coord, strength=strength, segment=segment, name=self.get_data_name(idx))

    def get_data_name(self, idx):
        p = self.data_list[idx % len(self.data_list)]
        seq = os.path.basename(os.path.dirname(os.path.dirname(p)))
        frame = os.path.splitext(os.path.basename(p))[0]
        return f"{seq}_{frame}"
