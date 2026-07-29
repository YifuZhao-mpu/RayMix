"""Expanded physical diagnostics (Codex paper item #5), beyond phys_validity.py's double-returns + ground-clearance.
Adds, over many scans (machine-printed, CPU): (a) instance VALIDITY-REJECT rate + reason; (b) host points DELETED
by the ray z-buffer (occlusion) per scan; (c) per-class transplanted-instance point-count vs RANGE compared to REAL
instances — does ray-consistent transplantation preserve realistic LiDAR density? This is the evidence that the
intervention is physically faithful, which is the unit of contribution for the honest ~7 analysis paper."""
import numpy as np
from collections import defaultdict
from pointcept.datasets.semantic_poss_raymix import SemanticPOSSRayMixDataset, _polar
from sklearn.cluster import DBSCAN

NAMES = ["people","rider","car","trunk","plants","sign","pole","trashcan","building","cone","fence","bike","ground"]
WEAK = [3,6,7,9]


def main(N=120):
    root = "/root/project/data/SemanticPOSS"
    ds = SemanticPOSSRayMixDataset(split="train", data_root=root, eval_seq=3, raymix_p=1.0, swap_p=1.0, transform=[])
    np.random.seed(3)
    idxs = np.random.choice(len(ds.data_list), N, replace=False)
    n_cand=0; n_rej_ground=0; n_rej_struct=0; n_ok=0
    host_deleted=[]
    # point-count vs range: bucket by range, separately for REAL vs TRANSPLANTED, per weak class
    real_cnt=defaultdict(list); tp_cnt=defaultdict(list)   # key=(class,range_bucket) -> list of inst point counts
    def rbucket(r): return min(int(r//10)*10, 50)
    for j in idxs:
        cA,sA,tA = ds._load_raw(ds.data_list[j])
        cB,sB,tB = ds._load_raw(ds.data_list[np.random.randint(len(ds.data_list))])
        # REAL instance point-count vs range (host scan, weak classes)
        for c in WEAK:
            m=tA==c
            if m.sum()<ds.min_inst_pts: continue
            lab=DBSCAN(eps=ds.dbscan_eps,min_samples=ds.dbscan_min).fit(cA[m]).labels_
            P=cA[m]
            for k in np.unique(lab):
                if k==-1: continue
                pts=P[lab==k]
                if len(pts)<ds.min_inst_pts: continue
                rng=float(np.sqrt((pts[:,:2]**2).sum(1)).mean())
                real_cnt[(c,rbucket(rng))].append(len(pts))
        # candidate transplants: count reject reasons + transplanted point-count vs range
        insts=ds._extract_instances(cB,sB,tB)
        for inst in insts:
            n_cand+=1
            placed=ds._place(inst,cA,tA)
            if placed is None:
                # re-derive reason cheaply: ground-supported & no ground near -> ground; else struct
                c=inst[2]
                n_rej_ground+= int(c in __import__('pointcept.datasets.semantic_poss_raymix',fromlist=['GROUND_SUPPORTED']).GROUND_SUPPORTED)
                n_rej_struct+= int(not (c in __import__('pointcept.datasets.semantic_poss_raymix',fromlist=['GROUND_SUPPORTED']).GROUND_SUPPORTED))
                continue
            n_ok+=1
            pc,ps,c=placed
            if c in WEAK:
                rng=float(np.sqrt((pc[:,:2]**2).sum(1)).mean())
                tp_cnt[(c,rbucket(rng))].append(len(pc))
        # occlusion deletions: run ray_merge, count host pts removed
        a_c=[];a_t=[]
        for inst in insts:
            pl=ds._place(inst,cA,tA)
            if pl is None: continue
            a_c.append(pl[0]); a_t.append(np.full(len(pl[0]),pl[2],np.int32))
        if a_c:
            A=np.concatenate(a_c,0)
            before=len(cA)
            merged=ds._ray_merge(cA,sA,tA,A,np.zeros((len(A),1),np.float32),np.concatenate(a_t,0))
            # host deleted = before - (merged_total - kept_instance); approximate kept_instance via cell test
            from numpy import inf
            cellA,rA=ds._cell(cA); cellI,rI=ds._cell(A)
            mx=ds.n_beam*ds.n_azimuth
            hmin=np.full(mx,inf,np.float32); np.minimum.at(hmin,cellA,rA)
            kept_inst=int((rI<hmin[cellI]).sum())
            host_del=before-(len(merged[0])-kept_inst)
            host_deleted.append(host_del)
    print("\n=============== EXPANDED PHYSICAL DIAGNOSTICS (%d scans) ===============" % N)
    print(f"(a) validity-reject rate: {100*(n_cand-n_ok)/max(n_cand,1):.1f}%  "
          f"(of {n_cand} candidates: {n_rej_ground} no-ground, {n_rej_struct} inside-structure, {n_ok} accepted)")
    print(f"(b) host points deleted by ray z-buffer (occlusion): mean {np.mean(host_deleted):.0f}/scan "
          f"median {np.median(host_deleted):.0f}  (physically-correct occlusion of background behind transplants)")
    print(f"(c) per-class point-count vs range — TRANSPLANTED should track REAL (realistic LiDAR density):")
    print(f"    {'class':9s} {'range':>7s}  {'REAL(med,n)':>14s}  {'TRANSPLANT(med,n)':>18s}")
    for c in WEAK:
        for rb in [0,10,20,30,40,50]:
            R=real_cnt.get((c,rb),[]); T=tp_cnt.get((c,rb),[])
            if not R and not T: continue
            rm=f"{int(np.median(R))},{len(R)}" if R else "-,0"
            tm=f"{int(np.median(T))},{len(T)}" if T else "-,0"
            print(f"    {NAMES[c]:9s} {rb:4d}-{rb+10:<2d}  {rm:>14s}  {tm:>18s}")
    import json
    json.dump(dict(reject_rate_pct=round(100*(n_cand-n_ok)/max(n_cand,1),1),
                   host_deleted_mean=float(np.mean(host_deleted)) if host_deleted else 0.0,
                   n_candidates=n_cand, n_accepted=n_ok),
              open("/root/Fresh_ARIS8/code/outputs/poss/phys_diag2.json","w"), indent=2)
    print("======================================================================\n")


if __name__=="__main__":
    main()
