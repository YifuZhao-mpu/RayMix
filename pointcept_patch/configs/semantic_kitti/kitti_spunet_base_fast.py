_base_ = ["../_base_/default_runtime.py"]
# SemanticKITTI 64-beam SpUNet baseline — REDUCED-protocol (24 ep) controlled cross-sensor SANITY for RayMix paper.
# All three arms (baseline / PolarMix-naive / RayMix) share this recipe; only the train dataset/aug differs.
batch_size = 12
mix_prob = 0.8
empty_cache = False
enable_amp = True
enable_wandb = False
seed = 0
num_classes = 19
ignore_index = -1

model = dict(
    type="DefaultSegmentor",
    backbone=dict(type="SpUNet-v1m1", in_channels=4, num_classes=num_classes,
                  channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2)),
    criteria=[dict(type="CrossEntropyLoss",
                   weight=[3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704, 10.1922, 1.6155, 4.2187,
                           1.9385, 5.5455, 2.0198, 2.6261, 1.3212, 5.1102, 2.5492, 5.8585, 7.3929],
                   loss_weight=1.0, ignore_index=ignore_index)],
)
epoch = 24
eval_epoch = 24
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(type="OneCycleLR", max_lr=optimizer["lr"], pct_start=0.04,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)

dataset_type = "SemanticKITTIDataset"
data_root = "/datasets/data/semiantic_kitti"
names = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",
         "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
         "terrain", "pole", "traffic-sign"]

_clip = dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2))
_tf_train = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    _clip,
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength")),
]
_tf_val = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    _clip,
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
         feat_keys=("coord", "strength")),
]
data = dict(
    num_classes=num_classes, ignore_index=ignore_index, names=names,
    train=dict(type=dataset_type, split="train", data_root=data_root, transform=_tf_train,
               test_mode=False, ignore_index=ignore_index),
    val=dict(type=dataset_type, split="val", data_root=data_root, transform=_tf_val,
             test_mode=False, ignore_index=ignore_index),
    test=dict(type=dataset_type, split="val", data_root=data_root,
              transform=[_clip, dict(type="Copy", keys_dict={"segment": "origin_segment"}),
                         dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_inverse=True)],
              test_mode=True,
              test_cfg=dict(voxelize=dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="test",
                                          return_grid_coord=True),
                            crop=None,
                            post_transform=[dict(type="ToTensor"),
                                            dict(type="Collect", keys=("coord", "grid_coord", "index"),
                                                 feat_keys=("coord", "strength"))],
                            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]]),
              ignore_index=ignore_index),
)
