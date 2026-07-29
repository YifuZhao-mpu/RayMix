_base_ = ["../_base_/default_runtime.py"]
# SemanticPOSS from-scratch baseline (MinkUNet/SpUNet via spconv) — beat-SOTA pilot arm A.
# Official 13-class protocol, train 00-05 exc 03, val 03. Target to beat: FRNet 53.5 mIoU.
batch_size = 12
mix_prob = 0.8
empty_cache = False
enable_amp = True
num_classes = 13
ignore_index = -1
enable_wandb = False
seed = 0

model = dict(
    type="DefaultSegmentor",
    backbone=dict(type="SpUNet-v1m1", in_channels=4, num_classes=num_classes,
                  channels=(32, 64, 128, 256, 256, 128, 96, 96), layers=(2, 3, 4, 6, 2, 2, 2, 2)),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
              dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index)],
)
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(type="OneCycleLR", max_lr=optimizer["lr"], pct_start=0.04,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)

dataset_type = "SemanticPOSSDataset"
data_root = "/root/project/data/SemanticPOSS"
names = ["people", "rider", "car", "trunk", "plants", "traffic-sign", "pole",
         "trashcan", "building", "cone-stone", "fence", "bike", "ground"]

_tf_train = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
    dict(type="SphereCrop", sample_rate=0.8, mode="random"),
    dict(type="SphereCrop", point_max=120000, mode="random"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength")),
]
_tf_val = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train",
         return_grid_coord=True, return_inverse=True),
    dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
         feat_keys=("coord", "strength")),
]
data = dict(
    num_classes=num_classes, ignore_index=ignore_index, names=names,
    train=dict(type="SemanticPOSSPolarMixDataset", split="train", data_root=data_root, transform=_tf_train,
               test_mode=False, ignore_index=ignore_index, eval_seq=3),
    val=dict(type=dataset_type, split="val", data_root=data_root, transform=_tf_val,
             test_mode=False, ignore_index=ignore_index, eval_seq=3),
    test=dict(type=dataset_type, split="val", data_root=data_root,
              transform=[dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
                         dict(type="Copy", keys_dict={"segment": "origin_segment"}),
                         dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train", return_inverse=True)],
              test_mode=True,
              test_cfg=dict(voxelize=dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="test",
                                          return_grid_coord=True),
                            crop=None,
                            post_transform=[dict(type="ToTensor"),
                                            dict(type="Collect", keys=("coord", "grid_coord", "index"),
                                                 feat_keys=("coord", "strength"))],
                            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]]),
              ignore_index=ignore_index, eval_seq=3),
)
