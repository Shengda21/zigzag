"""Cross-check new derived values against archived independent paper outputs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "capability-zigzag-repro"
ours = json.loads((HERE / "derived_results.json").read_text(encoding="utf-8"))["arms"]
event = json.loads((HERE / "event_bootstrap.json").read_text(encoding="utf-8"))["arms"]
old_sweep = json.loads((REPO / "paper_sec4/theta_sweep.json").read_text(encoding="utf-8"))
old_mv = json.loads((REPO / "paper_sec4/mv_counts.json").read_text(encoding="utf-8"))["arms"]
assert len(ours) == len(event) == len(old_sweep) == len(old_mv) == 6
for a, e, s, mv in zip(ours, event, old_sweep, old_mv):
    assert a["run"] == e["run"] == s["run"] == mv["run"]
    assert a["E1"]["B_history"]["final_alpha1"] == mv["thetas"]["0.8"]["K_B"]["hist_union_final"]["spurious"]
    assert abs(a["E2"]["sup_error"]["all_cells"]["q95"] - s["bootstrap_eps"]["sup_q95"]) < 1e-12
    for pair in ("288-304", "480-496", "496-512"):
        i = a["steps"].index(int(pair.split("-")[0]))
        for delta in ("0.0", "0.02", "0.05", "0.1"):
            point = e["pairs"][pair][delta]["point_counts_birth_death_transition"]
            archived = s["robust_counts"]["K_B"]["0.8"][delta]["H1"][i]
            assert point == archived, (a["run"], pair, delta, point, archived)
print("Verified six arms: B artifact dimensions, full-table max-error quantiles, and all 72 fixed-bridge point counts match archived outputs.")
