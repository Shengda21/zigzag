"""Compare freshly computed results (results/) with the archived outputs shipped in the repository.
Compare temporal statistics against the corrected references where available."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
ENDPOINT_AUDIT = ROOT / "followup/global_endpoint_audit"
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
            corrected = ENDPOINT_AUDIT / f.name.replace(".robust.json", ".corrected.robust.json")
            legacy_temporal = "temporal_S3" in a and "temporal_S3" in b
            if name in {"forget3_confirm2_SFT", "pythia_runs_55"}:
                reference = json.loads(corrected.read_text(encoding="utf8"))
                b["temporal_S3"] = reference["temporal_S3"]
                legacy_temporal = False
            keys = [key for key in KEYS if key in a and key in b
                    and (key != "temporal_S3" or not legacy_temporal)]
            differences = [key for key in keys if a[key] != b[key]]
            status = ("DIFFERS in " + ",".join(differences) if differences
                      else "IDENTICAL on " + ",".join(keys))
            note = " (archived temporal_S3 uses the legacy statistic)" if legacy_temporal else ""
            print(f"{name}/{f.name}: {status}{note}")
    for name, new, old in [("pythia_h0_bars_55", RES / "pythia_h0_bars_55.json", ROOT / "paper_sec4/pythia_h0_bars_55.json")]:
        if new.exists():
            a, b = json.loads(new.read_text(encoding="utf8")), json.loads(old.read_text(encoding="utf8"))
            print(f"{name}: counts {'identical' if a['counts'] == b['counts'] else 'DIFFER'}; zigzag bars {'identical' if a['zigzag'] == b['zigzag'] else 'DIFFER'}")


if __name__ == "__main__":
    main()
