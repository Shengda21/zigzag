"""Independent brute-force chain-space check of every real-data certificate.

Uses explicit enumeration of all edge chains and XOR sums, with no homology
representatives, induced-map coordinates, or Gaussian rank from cw_zigzag.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import certify as C

HERE = Path(__file__).resolve().parent
_, sub = C.bcomplex()
cells = sub.cells
index = {n: i for i, (n, _, _) in enumerate(cells)}
edges = [i for i, (_, d, _) in enumerate(cells) if d == 1]
faces = [i for i, (_, d, _) in enumerate(cells) if d == 2]
boundary = [sum(1 << index[b] for b in bs) for _, _, bs in cells]


@lru_cache(None)
def cycles(mask):
    active = [i for i in edges if mask >> i & 1]
    chains = [(0, 0)]
    for i in active:
        chains += [(z ^ (1 << i), b ^ boundary[i]) for z, b in chains[:]]
    return frozenset(z for z, b in chains if b == 0)


@lru_cache(None)
def boundaries(mask):
    z = {0}
    for i in faces:
        if mask >> i & 1:
            z |= {x ^ boundary[i] for x in z}
    return frozenset(z)


def add(A, B):
    return {a ^ b for a in A for b in B}


@lru_cache(None)
def direct_kappa(lbridge, ubridge, sources):
    J = boundaries(ubridge)
    for s in sources:
        J = add(J, cycles(s))
    IJ = add(J, cycles(lbridge))
    assert len(IJ) % len(J) == 0
    ratio = len(IJ) // len(J)
    assert ratio & (ratio - 1) == 0
    return ratio.bit_length() - 1


def masks(table, theta):
    # Recursive face closure, independently from R.Masker.
    raw = table >= theta - 1e-12
    out=[]
    for t in range(table.shape[1]):
        present = {}
        def has(i):
            if i not in present:
                present[i] = bool(raw[i,t]) and all(has(index[b]) for b in cells[i][2])
            return present[i]
        out.append(sum(1<<i for i in range(len(cells)) if has(i)))
    return out


def main():
    data=json.loads((HERE / "real_results.json").read_text(encoding="utf-8"))
    checks=0
    for arm in data["arms"]:
        for method,res in arm["methods"].items():
            arr=np.load(HERE / f'{arm["run"]}_{method}.npz')
            for mode in ("coordinate_aware","scalar_compression"):
                if mode == "coordinate_aware":
                    L,U=arr["lower"],arr["upper"]
                else:
                    eps=res["global_epsilon"]
                    L,U=np.maximum(0,arr["point"]-eps),np.minimum(1,arr["point"]+eps)
                for theta in C.THETAS:
                    for delta in C.DELTAS:
                        lo,up=masks(L,theta+delta),masks(U,theta-delta)
                        reported=res[mode][str(theta)][str(delta)]
                        ranks=[]
                        for a,b in zip(lo,up):
                            ranks.append(direct_kappa(a,b,()))
                            checks+=1
                        assert ranks==reported["checkpoint_image_rank_lower"]
                        records=[]
                        for t in range(len(lo)-1):
                            lw,uw=lo[t]|lo[t+1],up[t]|up[t+1]
                            vals=[direct_kappa(lw,uw,ss) for ss in ((up[t],),(up[t+1],),(up[t],up[t+1]))]
                            checks+=3
                            if any(vals):
                                records.append(dict(steps=arm["steps"][t:t+2],lower_kappa_forward_backward_transition=vals))
                        assert records==reported["positive_transition_records"]
    C.dump("independent_validation.json",dict(checked_chain_space_ranks=checks,
        unique_rank_configurations=direct_kappa.cache_info().currsize,
        method="Exhaustive edge-chain and XOR-span enumeration; independent of CW homology/induced-map/rank implementation.",status="PASS"))
    print("PASS",checks,"rank equalities")


if __name__=="__main__":
    main()
