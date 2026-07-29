_base_ = ["../_base_/default_runtime.py"]
# SemanticKITTI 64-beam PT-v3 (transformer) + RayMix — SECOND-BACKBONE generalization arm for the RayMix paper.
# Goal: test whether the "physical validity yields no robust converged gain" result holds on a transformer
# backbone (PT-v3), not just sparse-voxel SpUNet (closes kill-argument P_6 scope).
# DATA PIPELINE + RayMix dataset are IDENTICAL to kitti_spunet_raymix_fast.py (only the backbone/recipe differ,
# and the recipe is PT-v3's own tuned one — shared across both PT-v3 arms so the RayMix-vs-balnaive DELTA is clean).
# balanced-naive (no-phys) arm = launch with:
#   --options data.train.snap_ground=False data.train.validity=False data.train.use_ray_merge=False
batch_size = 4          # PT-v3 on 64-beam KITTI is heavier than SpUNet; tuned by smoke test (1 GPU)
mix_prob = 0.8
empty_cache = False
enable_amp = True
enable_wandb = False
seed = 0
num_classes = 19
ignore_index = -1

model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
)
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(type="OneCycleLR", max_lr=[0.002, 0.0002], pct_start=0.04,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
param_dicts = [dict(keyword="block", lr=0.0002)]

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
    train=dict(type="SemanticKITTIRayMixDataset", split="train", data_root=data_root, transform=_tf_train,
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
