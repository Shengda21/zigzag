"""Check sampling/trajectory assumptions against archived design and outputs."""
import hashlib
import json
from pathlib import Path

import certify as C

def main():
    files={"train":C.REPO / "stage_ab/code/zz_train.py",
           "run":C.REPO / "complex_worlds/code/run_complex.py"}
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    results=[]
    for run in sorted(C.RUNS.iterdir()):
        if not run.is_dir() or not any(run.glob("eval_*.jsonl*")):
            continue
        rm=json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        assert all(hashes[k]==rm["code_sha256"][k] for k in hashes)
        wm=json.loads((C.REPO / f'complex_worlds/data/RCN12_FORGET3/world{rm["world"]}/MANIFEST.json').read_text(encoding="utf-8"))
        split=wm["split"]
        assert (len(split["train"]),len(split["structure"]),len(split["independent"]))==(960,32,32)
        st=[tuple(x) for x in split["structure"]]
        other={st[j]:st[(j+1)%32] for j in range(32)}
        ev=C.ZA.load_run(run)
        assert sorted(ev)==list(range(0,769,16))
        paired_rows=0
        for rows in ev.values():
            for r in rows:
                if r.get("kind")=="joint" and r.get("override") is True:
                    assert tuple(r["other"])==other[tuple(r["root"])]
                    paired_rows+=1
        results.append(dict(run=run.name,source_code_matches_manifest=True,
            split_counts={k:len(v) for k,v in split.items()},
            fixed_grid_checkpoints=len(ev),paired_rows_cyclic_mapping_verified=paired_rows,
            recorded_save_every=rm["save_every"],local_weight_directories=[p.name for p in run.iterdir() if p.is_dir()]))
    C.dump("design_audit.json",dict(arms=results,status="PASS",
        source_semantics="Fixed-budget training; step-only lr; eval saves/restores RNG and does not select updates/checkpoints. Randomized-design inference conditions on fixed codebooks, training-root set and trajectory."))
    print("PASS",len(results),"arm sampling/trajectory audits")

if __name__=="__main__":
    main()
