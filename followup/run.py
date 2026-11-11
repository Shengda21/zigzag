"""Run unchanged archived analysis with one explicit runtime path relocation."""
import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("script", help="e.g. e0_e3/verify_derived.py")
args = parser.parse_args()
path = (ROOT / "followup" / args.script).resolve()
if path.parent.parent != ROOT / "followup" or path.suffix != ".py":
    parser.error("choose a Python script directly inside a followup analysis directory")
sys.path.insert(0, str(path.parent))
sys.dont_write_bytecode = True

def execute(file, name):
    source = file.read_text(encoding="utf-8")
    # Resolve data paths relative to the current checkout.
    lines = source.splitlines(keepends=True)
    source = "".join("REPO = Path(" + repr(str(ROOT)) + ")\n"
                     if line.startswith("REPO = ") else line for line in lines)
    module = types.ModuleType(name)
    module.__file__ = str(file)
    sys.modules[name] = module
    exec(compile(source, str(file), "exec"), module.__dict__)
    return module

if path.parent.name == "confidence" and path.name != "certify.py":
    execute(path.parent / "certify.py", "certify")
execute(path, "__main__")
