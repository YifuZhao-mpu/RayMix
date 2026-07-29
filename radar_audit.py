"""Radar-support audit (Codex's validate-or-kill for Doppler-DITR), on LOCAL nuScenes (no download, no labels).
Tests the riskiest premise — does nuScenes radar give usable dynamic-thing support where LiDAR is hard? — which
my own prior finding flags as doubtful (radar too sparse, ~1% moving, 2D z=0). MEASURE, don't assume.

For ~N keyframes, every dynamic-thing GT box:
  hard = center range >25m OR <40 LiDAR pts inside; moving = GT speed >1 m/s.
  radar coverage = >=1 radar return inside the (z-expanded) box; Doppler = associated |v_comp|.
Machine-prints: radar pts/keyframe; per-class coverage (cars vs non-car); coverage hard-vs-easy;
Doppler separation moving-vs-static (AUROC); static-bg false-support. Codex kill: coverage mostly cars OR
moving-vs-static AUROC weak OR hard-instance coverage tiny."""
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

DR = '/datasets/data/nuscenes'
CAT = {'car':'car','truck':'truck','bus':'bus','trailer':'trailer','construction':'construction',
       'motorcycle':'motorcycle','bicycle':'bicycle','pedestrian':'pedestrian'}

def cat_of(name):
    for k,v in CAT.items():
        if k in name: return v
    return None

def to_global(pc_xyz, sd, nusc):
    """sensor-frame Nx3 -> global, via calibrated_sensor then ego_pose."""
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    p = pc_xyz.copy()
    p = (Quaternion(cs['rotation']).rotation_matrix @ p.T).T + np.array(cs['translation'])
    p = (Quaternion(ep['rotation']).rotation_matrix @ p.T).T + np.array(ep['translation'])
    return p

def main(N=400):
    nusc = NuScenes(version='v1.0-trainval', dataroot=DR, verbose=False)
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(nusc.sample), min(N, len(nusc.sample)), replace=False)
    radars = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
    radar_per_kf=[]
    # per-class: [n_instances, n_hard, n_cov_all, n_cov_hard]
    from collections import defaultdict
    pc_stat=defaultdict(lambda: np.zeros(4))
    mov_vcomp=[]; sta_vcomp=[]      # |v_comp| of radar in moving vs static dyn-thing boxes (Doppler sep)
    hard_cov=np.zeros(2); easy_cov=np.zeros(2)  # [covered, total]
    for ii in idxs:
        s=nusc.sample[int(ii)]
        ld=nusc.get('sample_data', s['data']['LIDAR_TOP'])
        lp=LidarPointCloud.from_file(f"{DR}/{ld['filename']}")
        lg=to_global(lp.points[:3].T, ld, nusc)   # lidar global Nx3
        # radar global
        rad_g=[]; rad_v=[]
        for r in radars:
            sd=nusc.get('sample_data', s['data'][r])
            rp=RadarPointCloud.from_file(f"{DR}/{sd['filename']}")
            if rp.points.shape[1]==0: continue
            rad_g.append(to_global(rp.points[:3].T, sd, nusc))
            rad_v.append(np.sqrt(rp.points[8]**2+rp.points[9]**2))
        rad_g=np.concatenate(rad_g,0) if rad_g else np.zeros((0,3))
        rad_v=np.concatenate(rad_v,0) if rad_v else np.zeros((0,))
        radar_per_kf.append(len(rad_g))
        ego=nusc.get('ego_pose', ld['ego_pose_token'])['translation']
        for ann_tok in s['anns']:
            a=nusc.get('sample_annotation', ann_tok)
            c=cat_of(a['category_name'])
            if c is None: continue
            box=nusc.get_box(ann_tok)               # global frame
            ctr=box.center
            rng_=np.linalg.norm(ctr[:2]-np.array(ego)[:2])
            # lidar pts in box
            lin=points_in_box(box, lg.T)
            nlid=int(lin.sum())
            hard = (rng_>25.0) or (nlid<40)
            # radar in box (z-expanded since radar z is unreliable/0)
            covered=False; vc_in=[]
            if len(rad_g):
                b2=nusc.get_box(ann_tok); b2.wlh=b2.wlh*np.array([1.3,1.3,4.0])  # expand, esp z
                rin=points_in_box(b2, rad_g.T)
                covered=bool(rin.sum()>0)
                if covered: vc_in=rad_v[rin]
            v=nusc.box_velocity(ann_tok); gsp=np.linalg.norm(v[:2]) if not np.isnan(v[0]) else 0.0
            st=pc_stat[c]; st[0]+=1; st[1]+=hard; st[2]+=covered; st[3]+= (covered and hard)
            (hard_cov if hard else easy_cov)[1]+=1
            (hard_cov if hard else easy_cov)[0]+= covered
            if covered:
                (mov_vcomp if gsp>1.0 else sta_vcomp).extend(list(vc_in))
    # ---- report ----
    print("\n==================== RADAR-SUPPORT AUDIT (%d keyframes) ====================" % len(idxs))
    print("radar pts/keyframe: mean %.0f  median %.0f   (LiDAR ~34k => ~%.2f%% density)" %
          (np.mean(radar_per_kf), np.median(radar_per_kf), 100*np.mean(radar_per_kf)/34000))
    print("\nper-class dynamic-thing radar coverage (>=1 radar return inside box):")
    print(f"  {'class':13s} {'#inst':>6s} {'%hard':>6s} {'cov_all%':>8s} {'cov_HARD%':>9s}")
    car_cov=[]; noncar_cov=[]
    for c,st in sorted(pc_stat.items(), key=lambda x:-x[1][0]):
        n,nh,ca,ch=st
        cov_all=100*ca/max(n,1); cov_hard=100*ch/max(nh,1)
        print(f"  {c:13s} {int(n):6d} {100*nh/max(n,1):5.0f}% {cov_all:7.1f}% {cov_hard:8.1f}%")
        (car_cov if c=='car' else noncar_cov).append((ca,n))
    nc_ca=sum(x[0] for x in noncar_cov); nc_n=sum(x[1] for x in noncar_cov)
    print(f"\n  NON-CAR dynamic things overall coverage: {100*nc_ca/max(nc_n,1):.1f}%  (n={int(nc_n)})")
    print(f"  HARD-instance coverage (all dyn things): {100*hard_cov[0]/max(hard_cov[1],1):.1f}%  (n_hard={int(hard_cov[1])})")
    print(f"  easy-instance coverage:                  {100*easy_cov[0]/max(easy_cov[1],1):.1f}%")
    # Doppler separation moving vs static (AUROC of |v_comp|)
    mov=np.array(mov_vcomp); sta=np.array(sta_vcomp)
    if len(mov)>5 and len(sta)>5:
        from itertools import product
        # fast AUROC via rank
        allv=np.concatenate([mov,sta]); lab=np.concatenate([np.ones(len(mov)),np.zeros(len(sta))])
        order=np.argsort(allv); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(allv)+1)
        auroc=(ranks[lab==1].sum()-len(mov)*(len(mov)+1)/2)/(len(mov)*len(sta))
        print(f"\n  Doppler |v_comp| separation moving-vs-static instances: AUROC={auroc:.3f}  "
              f"(moving radar med {np.median(mov):.2f} vs static {np.median(sta):.2f} m/s)")
    print("\n  CODEX KILL CRITERIA: coverage mostly cars / hard-coverage tiny / AUROC ~0.5 => radar won't beat DITR.")
    print("============================================================================\n")
    import json
    json.dump(dict(radar_per_kf_mean=float(np.mean(radar_per_kf)),
                   noncar_cov_pct=float(100*nc_ca/max(nc_n,1)),
                   hard_cov_pct=float(100*hard_cov[0]/max(hard_cov[1],1)),
                   per_class={c:[float(x) for x in st] for c,st in pc_stat.items()}),
              open("/root/Fresh_ARIS8/code/outputs/poss/radar_audit.json","w"), indent=2)

if __name__=='__main__':
    main()
