"""Print absolute CoT vs no-CoT accuracies (and the gap no-CoT costs) for
GSM8K and MoreHopQA, plus intervention accuracies for prompt-repeat and
filler/restricted-CoT conditions. Reuses the eval loaders from analysis.py."""

import json
import zipfile
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import samples_df

SPECIAL_LOGS = Path(__file__).resolve().parent.parent / "special-logs"


def find_eval_file(directory: Path, name_pattern: str) -> Path | None:
    candidates = [f for f in directory.glob("*.eval") if name_pattern in f.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def acc_gsm8k(p):
    return float(samples_df(logs=str(p), quiet=True)["score_gsm8k_scorer"].mean())


def acc_morehop(p):
    return float(samples_df(logs=str(p), quiet=True)["score_morehopqa_scorer"].mean())


def acc_morehop_zip(p):
    vals = []
    with zipfile.ZipFile(p, "r") as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with z.open(name) as f:
                    s = json.load(f)
                v = s.get("scores", {}).get("morehopqa_scorer", {}).get("value")
                if v is not None:
                    vals.append(v)
    return sum(vals) / len(vals)


def report(title, cot_acc, nocot_acc, interventions):
    gap = cot_acc - nocot_acc
    print(f"\n=== {title} ===")
    print(f"  CoT (ceiling)   : {cot_acc:.3f}")
    print(f"  no-CoT (floor)  : {nocot_acc:.3f}")
    print(f"  ABSOLUTE DROP   : {gap:+.3f}  ({gap*100:.1f} pts)  <- how much no-CoT hurt")
    for label, acc in interventions:
        rec = (acc - nocot_acc) / gap if gap > 0 else float("nan")
        print(f"  {label:14s}: acc={acc:.3f}  (+{acc-nocot_acc:+.3f} vs floor, recovery={rec:.3f})")


GPT52 = SPECIAL_LOGS / "gpt52"

# ---------- GSM8K prompt repeats ----------
gsm8k_models = [
    ("GPT-5.2 GSM8K (repeats)", GPT52, "gpt52-gsm8k-cot-subset", "gpt52-gsm8k-repeat1-subset",
     {2: "gpt52-gsm8k-repeat2-subset", 3: "gpt52-gsm8k-repeat3-subset",
      5: "gpt52-gsm8k-repeat5-subset", 10: "gpt52-gsm8k-repeat10-subset"}),
    ("Opus 4.5 GSM8K (repeats)", SPECIAL_LOGS, "opus4.5-gsm8k-cot-subset", "opus4.5-gsm8k-no-cot-subset",
     {2: "opus4.5-gsm8k-repeat2-subset", 3: "opus4.5-gsm8k-repeat3-subset",
      5: "opus4.5-gsm8k-repeat5-subset", 10: "opus4.5-gsm8k-repeat10-subset"}),
    ("Qwen2.5-7B GSM8K (repeats)", SPECIAL_LOGS, "qwen2.5-7b-gsm8k-cot-subset", "qwen2.5-7b-gsm8k-repeat1-subset",
     {2: "qwen2.5-7b-gsm8k-repeat2-subset", 3: "qwen2.5-7b-gsm8k-repeat3-subset",
      5: "qwen2.5-7b-gsm8k-repeat5-subset", 10: "qwen2.5-7b-gsm8k-repeat10-subset"}),
]
for title, d, cotp, nocotp, reps in gsm8k_models:
    cp, np_ = find_eval_file(d, cotp), find_eval_file(d, nocotp)
    if not cp or not np_:
        print(f"\n=== {title} === (missing cot/no-cot)")
        continue
    interv = []
    for n, pat in reps.items():
        fp = find_eval_file(d, pat)
        if fp:
            interv.append((f"repeat{n}", acc_gsm8k(fp)))
    report(title, acc_gsm8k(cp), acc_gsm8k(np_), interv)

# ---------- GSM8K filler tokens (Qwen2.5-14B) ----------
d = SPECIAL_LOGS
cp = find_eval_file(d, "qwen2.5-14b-gsm8k-cot-subset") or find_eval_file(d, "qwen14b-gsm8k-cot-subset")
np_ = find_eval_file(d, "qwen2.5-14b-gsm8k-no-cot-subset")
if cp and np_:
    interv = []
    for lbl, pat in [("filler10", "qwen2.5-14b-gsm8k-filler10-subset"),
                     ("filler100", "qwen2.5-14b-gsm8k-filler100-subset"),
                     ("filler1000", "qwen2.5-14b-gsm8k-filler1000-subset")]:
        fp = find_eval_file(d, pat)
        if fp:
            interv.append((lbl, acc_gsm8k(fp)))
    report("Qwen2.5-14B GSM8K (filler tokens)", acc_gsm8k(cp), acc_gsm8k(np_), interv)

# ---------- MoreHopQA prompt repeats ----------
# GPT-5.2 morehop evals are zip-based
cp = find_eval_file(GPT52, "gpt52-morehopqa-cot-subset")
np_ = find_eval_file(GPT52, "gpt52-morehopqa-no-cot-subset")
if cp and np_:
    interv = []
    for n, pat in {2: "gpt52-morehopqa-repeat2-subset", 3: "gpt52-morehopqa-repeat3-subset",
                   5: "gpt52-morehopqa-repeat5-subset", 10: "gpt52-morehopqa-repeat10-subset"}.items():
        fp = find_eval_file(GPT52, pat)
        if fp:
            interv.append((f"repeat{n}", acc_morehop_zip(fp)))
    report("GPT-5.2 MoreHopQA (repeats)", acc_morehop_zip(cp), acc_morehop_zip(np_), interv)

cp = find_eval_file(SPECIAL_LOGS, "opus4.5-morehopqa-cot-subset")
np_ = find_eval_file(SPECIAL_LOGS, "opus4.5-morehopqa-no-cot-subset")
if cp and np_:
    interv = []
    for n, pat in {2: "opus4.5-morehopqa-repeat2-subset", 3: "opus4.5-morehopqa-repeat3-subset",
                   5: "opus4.5-morehopqa-repeat5-subset", 10: "opus4.5-morehopqa-repeat10-subset"}.items():
        fp = find_eval_file(SPECIAL_LOGS, pat)
        if fp:
            interv.append((f"repeat{n}", acc_morehop(fp)))
    report("Opus 4.5 MoreHopQA (repeats)", acc_morehop(cp), acc_morehop(np_), interv)

# ---------- MoreHopQA restricted-CoT / filler (max_len) ----------
for title, pats, accfn in [
    ("GPT-5.2 MoreHopQA (restricted CoT)",
     {"cot": "unconstrained-morehopqa-gpt-5.2", "no_cot": "no-cot-morehopqa-gpt-5.2",
      "max_len_3": "2stage-stripped-max-len-3-morehopqa-gpt-5.2",
      "max_len_4": "2stage-stripped-max-len-4-morehopqa-gpt-5.2",
      "max_len_5": "2stage-stripped-max-len-5-morehopqa-gpt-5.2"}, acc_morehop),
    ("Opus 4.5 MoreHopQA (restricted CoT)",
     {"cot": "unconstrained-morehopqa-claude-opus-4.5", "no_cot": "no-cot-morehopqa-claude-opus-4.5",
      "max_len_3": "2stage-stripped-max-len-3-morehopqa-claude-opus-4.5",
      "max_len_4": "2stage-stripped-max-len-4-morehopqa-claude-opus-4.5",
      "max_len_5": "2stage-stripped-max-len-5-morehopqa-claude-opus-4.5"}, acc_morehop),
]:
    cp, np_ = find_eval_file(SPECIAL_LOGS, pats["cot"]), find_eval_file(SPECIAL_LOGS, pats["no_cot"])
    if not cp or not np_:
        print(f"\n=== {title} === (missing cot/no-cot)")
        continue
    interv = []
    for k in ("max_len_3", "max_len_4", "max_len_5"):
        fp = find_eval_file(SPECIAL_LOGS, pats[k])
        if fp:
            interv.append((k, accfn(fp)))
    report(title, accfn(cp), accfn(np_), interv)
