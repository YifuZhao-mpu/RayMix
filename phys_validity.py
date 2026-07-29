"""Physical-validity diagnostic: do PolarMix's naive paste vs our ray-consistent transplant differ in
LiDAR physical plausibility of the ADDED instance points? Measures, per scan, over the same host+donor:
  (A) ray double-returns: instance points sharing a (beam,azimuth) ray cell with a host point at a
      DIFFERENT range (a physical impossibility for a single-return sensor) that the method failed to resolve;
  (B) ground-floating: for ground-supported classes, |instance base z - local host ground z| (should be ~0).
Reference = REAL instances measured in real scans. NO training, NO GPU. Machine-prints raw counts.
"""
import numpy as np
from pointcept.datasets.semantic_poss_polarmix import SemanticPOSSPolarMixDataset, _swap_sector
from pointcept.datasets.semantic_poss_raymix import SemanticPOSSRayMixDataset, _polar

NAMES = ["people","rider","car","trunk","plants","sign","pole","trashcan","building","cone","fence","bike","ground"]
GROUND_SUPPORTED = {1,3,5,6,7,9,11}
N_BEAM, N_AZ = 40, 1800
ELO, EHI = np.radians(-16.0), np.radians(8.0)


def cell_range(coord):
    r, theta, phi = _polar(coord)
    beam = np.clip(((phi-ELO)/(EHI-ELO)*N_BEAM).astype(np.int64), 0, N_BEAM-1)
    az = np.clip(((theta+np.pi)/(2*np.pi)*N_AZ).astype(np.int64), 0, N_AZ-1)
    return beam*N_AZ+az, r


def double_returns(scan_c, tol=0.5):
    """Honest final-scan metric: # of rays (beam,azimuth cells) that carry >1 return at ranges differing by
    > tol — a physical impossibility for a single-return spinning LiDAR. Applied to the FINAL scan a method
    produces, identically for both methods."""
    if len(scan_c)==0: return 0
    c, r = cell_range(scan_c)
    maxc = N_BEAM*N_AZ
    rmin = np.full(maxc, np.inf, np.float32); np.minimum.at(rmin, c, r)
    rmax = np.full(maxc, -np.inf, np.float32); np.maximum.at(rmax, c, r)
    return int(((rmax - rmin) > tol).sum())


def ground_float(host_c, host_t, add_c, add_t, radius=2.0):
    """For ground-supported added instances, |base_z - local host ground z|."""
    gm = host_t==12
    if not gm.any(): return []
    gz = host_c[gm]; devs=[]
    for c in np.unique(add_t):
        if int(c) not in GROUND_SUPPORTED: continue
        pts = add_c[add_t==c]
        if len(pts)<5: continue
        cen = pts[:,:2].mean(0); base = pts[:,2].min()
        near = gz[np.sum((gz[:,:2]-cen)**2,1)<radius**2]
        if len(near): devs.append(abs(float(base)-float(np.median(near[:,2]))))
    return devs


def main():
    root="/root/project/data/SemanticPOSS"
    pm = SemanticPOSSPolarMixDataset(split="train", data_root=root, eval_seq=3, polarmix_p=1.0, transform=[])
    rm = SemanticPOSSRayMixDataset(split="train", data_root=root, eval_seq=3, raymix_p=1.0, swap_p=1.0, transform=[])
    np.random.seed(7)
    n=30
    pm_dr=[]; rm_dr=[]; pm_gf=[]; rm_gf=[]; real_gf=[]
    for i in range(n):
        hi = i*40 % len(rm.data_list); di = (i*97+13) % len(rm.data_list)
        cH,sH,tH = rm._load_raw(rm.data_list[hi])
        cD,sD,tD = rm._load_raw(rm.data_list[di])
        host_base = double_returns(cH)   # host-alone conflicts (grid-discretization floor) -> subtract
        # ---- PolarMix naive instance paste (reproduce its instance branch on the SAME host/donor) ----
        inst = np.isin(tD, [0,1,2,3,6,7,9,11])
        if inst.sum()>0:
            ci,si,ti = cD[inst],sD[inst],tD[inst]
            add_c=[]; add_t=[]
            for th in (np.pi/2,np.pi,np.pi*3/2):
                co=np.cos(th); sn=np.sin(th)
                R=np.array([[co,-sn,0],[sn,co,0],[0,0,1]],np.float32)
                add_c.append(ci@R.T); add_t.append(ti)
            pm_add_c=np.concatenate(add_c,0); pm_add_t=np.concatenate(add_t,0)
            pm_final = np.concatenate([cH, pm_add_c],0)   # PolarMix does NOT resolve occlusion
            pm_dr.append(double_returns(pm_final) - host_base)
            pm_gf += ground_float(cH,tH, pm_add_c, pm_add_t)
        # ---- our ray-consistent transplant: extract+place+merge; measure the FINAL merged scan ----
        insts = rm._extract_instances(cD,sD,tD)
        a_c=[]; a_t=[]
        for it in insts:
            pl = rm._place(it, cH, tH)
            if pl is None: continue
            a_c.append(pl[0]); a_t.append(np.full(len(pl[0]), pl[2], np.int32))
        if a_c:
            add_all=np.concatenate(a_c,0); addt_all=np.concatenate(a_t,0)
            merged = rm._ray_merge(cH,sH,tH, add_all, np.zeros((len(add_all),1),np.float32), addt_all)
            rm_dr.append(double_returns(merged[0]) - host_base)   # final scan training actually sees
            # surviving added pts for ground-float stat
            hc,hr = cell_range(cH); ac,ar = cell_range(add_all)
            hmin=np.full(N_BEAM*N_AZ,np.inf,np.float32); np.minimum.at(hmin,hc,hr)
            keep = ar < hmin[ac]
            rm_gf += ground_float(cH,tH, add_all[keep], addt_all[keep])
        # ---- reference: REAL ground-supported instances in the host scan ----
        for c in GROUND_SUPPORTED:
            m=tH==c
            if m.sum()<20:continue
            from sklearn.cluster import DBSCAN
            lab=DBSCAN(eps=0.5,min_samples=10).fit(cH[m]).labels_
            for k in np.unique(lab):
                if k==-1:continue
                pts=cH[m][lab==k]
                if len(pts)<20:continue
                cen=pts[:,:2].mean(0); base=pts[:,2].min()
                gz=cH[tH==12]
                near=gz[np.sum((gz[:,:2]-cen)**2,1)<2.0**2]
                if len(near): real_gf.append(abs(float(base)-float(np.median(near[:,2]))))
    def stat(x): return f"mean {np.mean(x):.3f}  median {np.median(x):.3f}  n={len(x)}" if x else "n=0"
    print("\n==================== PHYSICAL-VALIDITY DIAGNOSTIC (30 scans) ====================")
    print("(A) ray double-returns per scan (added inst pts on a ray already occupied at a different range):")
    print(f"    PolarMix naive paste : mean {np.mean(pm_dr):.0f}  median {np.median(pm_dr):.0f}  (per scan)")
    print(f"    RayMix (ours, post z-buffer): mean {np.mean(rm_dr):.0f}  median {np.median(rm_dr):.0f}  <- should be ~0 by construction")
    print("\n(B) ground-floating |base_z - local ground z| (m), ground-supported classes:")
    print(f"    REAL instances      : {stat(real_gf)}")
    print(f"    PolarMix naive paste: {stat(pm_gf)}")
    print(f"    RayMix (ours)       : {stat(rm_gf)}")
    print("================================================================================\n")
    import json
    out=dict(double_returns=dict(polarmix_mean=float(np.mean(pm_dr)), raymix_mean=float(np.mean(rm_dr))),
             ground_float=dict(real_median=float(np.median(real_gf)) if real_gf else None,
                               polarmix_median=float(np.median(pm_gf)) if pm_gf else None,
                               raymix_median=float(np.median(rm_gf)) if rm_gf else None))
    open("/root/Fresh_ARIS8/code/outputs/poss/phys_validity.json","w").write(json.dumps(out,indent=2))
    print("wrote phys_validity.json")


if __name__=="__main__":
    main()
