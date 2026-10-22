"""Re-run the CPU-only analysis chain of the paper from the per-item evaluation files in this repository.

  python reproduce.py            # everything (about 60-90 minutes on a laptop; bootstrap B = 1000 and 1000 permutations)
  python reproduce.py --quick    # skip the long bootstrap steps (forget3/X2/grokking robust analysis, Pythia H0 bars); ~10 minutes
  python reproduce.py --only mv_counts theta_sweep figures

Outputs go to results/ (robust analyses) and paper_scripts/figures/ (figures); the shipped reference outputs are in
cloud_return_round4/analysis_r4, cloud_return_round5/analysis_r5, paper_sec4/*.json and paper_scripts/forget3_summary.json.
Run verify_against_archived.py afterwards to compare.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
ZZ = ROOT / "crosslingual" / "code" / "zz_robust.py"

STEPS = {
    # name: (long?, argv, env)
    "tests": (False, [PY, "-c", "import subprocess,sys,os; d=os.path.join(%r,'crosslingual','code'); "
                              "[subprocess.run([sys.executable, t], cwd=d, check=True) for t in ('test_cw_zigzag.py','test_zz_analyze.py')]" % str(ROOT)], {}),
    "forget3": (True, [PY, str(ZZ), "--line", "forget3", "--runs", str(ROOT / "cloud_return_round4/complex_worlds/runs/RCN12_FORGET3"),
                       "--design", str(ROOT / "complex_worlds/data/RCN12_FORGET3/world635/COMPLEX.json"), "--out", str(RES / "forget3_confirm2_SFT"),
                       "--boot", "1000", "--perm", "1000", "--procs", "3"], {}),
    "forget3_x2": (True, [PY, str(ZZ), "--line", "forget3", "--runs", str(ROOT / "cloud_return_round4/complex_worlds/runs/RCN12_FORGET3_X2"),
                          "--design", str(ROOT / "complex_worlds/data/RCN12_FORGET3_X2/world635/COMPLEX.json"), "--time-scale", "2",
                          "--out", str(RES / "forget3_confirm2_X2"), "--boot", "1000", "--perm", "1000", "--procs", "2"], {}),
    "grokking": (True, [PY, str(ZZ), "--line", "grok", "--runs", str(ROOT / "cloud_return_round2/grokking/runs_confirm"),
                        "--design", str(ROOT / "grokking/data/MODP/world705/COMPLEX.json"), "--out", str(RES / "grokking_confirm_posthoc"),
                        "--boot", "1000", "--perm", "1000", "--procs", "4"], {}),
    "pythia55": (False, [PY, str(ZZ), "--line", "public", "--runs", str(ROOT / "cloud_return_round5/public_ckpt/runs_merged"),
                         "--design", str(ROOT / "public_ckpt/spec/COMPLEX.json"), "--out", str(RES / "pythia_runs_55"), "--boot", "1000", "--perm", "1000"], {}),
    "pythia33": (False, [PY, str(ZZ), "--line", "public", "--runs", str(ROOT / "cloud_return_round4/public_ckpt/runs"),
                         "--design", str(ROOT / "public_ckpt/spec/COMPLEX.json"), "--out", str(RES / "pythia_runs_33"), "--boot", "1000", "--perm", "1000"], {}),
    "mv_counts": (False, [PY, str(ROOT / "paper_sec4" / "mv_counts.py")], {}),
    "theta_sweep": (False, [PY, str(ROOT / "paper_sec4" / "theta_sweep.py")], {}),
    "grok_tier2": (False, [PY, str(ROOT / "paper_sec4" / "check_grok_tier2.py")], {}),
    "pythia_h0_bars": (True, [PY, str(ROOT / "paper_sec4" / "pythia_h0_bars.py")],
                       {"ZZ_PYTHIA_RUN": str(ROOT / "cloud_return_round5/public_ckpt/runs_merged/pythia-1.4b"),
                        "ZZ_PYTHIA_H0_OUT": str(RES / "pythia_h0_bars_55.json")}),
    "summary": (False, [PY, str(ROOT / "paper_scripts" / "summarise_forget3.py")], {}),
    "numbers": (False, [PY, str(ROOT / "paper_scripts" / "fill_pythia_numbers.py"),
                        str(ROOT / "cloud_return_round5/analysis_r5/pythia_runs/pythia-1.4b.robust.json"),
                        str(ROOT / "paper_sec4/pythia_h0_bars_55.json"), str(RES / "pythia_numbers.tex")], {}),
    "figures": (False, [PY, str(ROOT / "paper_scripts" / "make_figures.py")],
                {"ZZ_PYTHIA_ROBUST": str(ROOT / "cloud_return_round5/analysis_r5/pythia_runs/pythia-1.4b.robust.json"),
                 "ZZ_PYTHIA_H0_JSON": str(ROOT / "paper_sec4/pythia_h0_bars_55.json")}),
}
ORDER = ["tests", "forget3", "forget3_x2", "grokking", "pythia55", "pythia33", "mv_counts", "theta_sweep", "grok_tier2", "pythia_h0_bars", "summary", "numbers", "figures"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the long bootstrap steps")
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of steps, in order: {ORDER}")
    a = ap.parse_args()
    names = a.only if a.only else ORDER
    for n in names:
        long_, argv, env = STEPS[n]
        if a.quick and long_:
            print(f"[{n}] skipped (--quick)")
            continue
        t0 = time.time()
        print(f"[{n}] running ...", flush=True)
        r = subprocess.run(argv, cwd=str(ROOT), env={**os.environ, **env})
        print(f"[{n}] exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
        if r.returncode:
            sys.exit(r.returncode)
    print("done; outputs in", RES)


if __name__ == "__main__":
    main()
