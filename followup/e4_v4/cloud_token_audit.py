"""Early tokenizer-only version of the mandatory E4 paired-budget preflight."""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

e4 = Path(sys.argv[1])
model = Path(sys.argv[2])
sys.path[:0] = [str(e4), str(e4 / "vendor")]
from generate import rows, dump, sha  # noqa: E402
import zz_train as T  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"
from generate import CONFIG
for world in CONFIG['pilot']:
    data = e4 / "data" / "pilot" / f"world{world}"
    pair = [T.encoded(tok, rows(data / f"{arm}.jsonl"), max_len=512, max_answer=96)
            for arm in ("edge", "joint")]
    budgets = [[(len(ids), sum(v != -100 for v in labels)) for ids, labels in enc] for enc in pair]
    mismatch = [i for i, (left, right) in enumerate(zip(*budgets)) if left != right]
    result={"world": world, "records": len(budgets[0]),
                      "per_record_token_mismatches": len(mismatch),
                      "first_mismatches": mismatch[:20],
                      "input_tokens": [sum(b[0] for b in arm) for arm in budgets],
                      "answer_tokens": [sum(b[1] for b in arm) for arm in budgets],
                      "prompt_tokens": [sum(b[0]-b[1] for b in arm) for arm in budgets],
                      "input_tokens_definition":"unpadded full training sequence, including answer",
                      "padding_counted":False,"tokenizer_sha256":sha(model/"tokenizer.json"),
                      "training_manifest_sha256":sha(data/"MANIFEST.json")}
    assert not mismatch,result
    dump(data/"TOKEN_AUDIT.json",result)
    print(json.dumps(result),flush=True)
