"""Compare freshly computed results (results/) with the archived outputs shipped in the repository.
All randomness is seeded (bootstrap seed 20260924/20260925, permutation seed +1), so the comparison is exact."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
PAIRS = [
    ("forget3_confirm2_SFT", RES / "forget3_confirm2_SFT", ROOT / "cloud_return_round4/analysis_r4/forget3_confirm2_SFT"),
    ("forget3_confirm2_X2", RES / "forget3_confirm2_X2", ROOT / "cloud_return_round4/analysis_r4/forget3_confirm2_X2"),
    ("grokking_confirm_posthoc", RES / "grokking_confirm_posthoc", ROOT / "cloud_return_round4/analysis_r4/grokking_confirm_posthoc"),
    ("pythia_runs_55", RES / "pythia_runs_55", ROOT / "cloud_return_round5/analysis_r5/pythia_runs"),
    ("pythia_runs_33", RES / "pythia_runs_33", ROOT / "cloud_return_round4/analysis_r4/pythia_runs"),
]
KEYS = ["steps", "scores_point", "point", "support", "robust", "P1", "P2", "temporal_S3"]


def main():
    for name, new_dir, old_dir in PAIRS:
        if not new_dir.exists():
            print(f"{name}: not computed (results missing)")
            continue
        for f in sorted(new_dir.glob("*.robust.json")):
            g = old_dir / f.name
            if not g.exists():
                print(f"{name}/{f.name}: no archived counterpart")
                continue
            a, b = json.loads(f.read_text(encoding="utf8")), json.loads(g.read_text(encoding="utf8"))
            diffs = [k for k in KEYS if k in a and k in b and a[k] != b[k]]
            print(f"{name}/{f.name}: {'IDENTICAL on ' + ','.join(k for k in KEYS if k in a) if not diffs else 'DIFFERS in ' + ','.join(diffs)}")
    for name, new, old in [("pythia_h0_bars_55", RES / "pythia_h0_bars_55.json", ROOT / "paper_sec4/pythia_h0_bars_55.json")]:
        if new.exists():
            a, b = json.loads(new.read_text(encoding="utf8")), json.loads(old.read_text(encoding="utf8"))
            print(f"{name}: counts {'identical' if a['counts'] == b['counts'] else 'DIFFER'}; zigzag bars {'identical' if a['zigzag'] == b['zigzag'] else 'DIFFER'}")


if __name__ == "__main__":
    main()
