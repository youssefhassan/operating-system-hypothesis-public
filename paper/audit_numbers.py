#!/usr/bin/env python3
"""Key-aware number audit for paper.md.

check_numbers.py only asks "does this value appear anywhere in the ledger?", ignoring sign
and key. This script asks the stronger question: for every numeric token in paper.md, is
there a row in NUMBER_MAP.md naming the ledger key it came from, and does that key's value
round (with sign) to the number as written?

NUMBER_MAP.md rows (pipe table, one per number as written in the draft):

    | number | key | note |

`number` is the token exactly as it appears in paper.md (e.g. -0.340, 64%, 5/6, 860).
`key` is a numbers_flat.json key, or `allow:<reason>` for design constants, or
`derived:<expression over keys>` (e.g. derived:100*exp03b.axes...share_of_association_explained)
which is evaluated. A number may appear several times in the draft; one row covers all.

Exit 1 on: a token with no row; a row whose key is missing from the ledger; a value that
does not round to the written number with the written sign.
"""
import json, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "paper.md")
flat = json.load(open(os.path.join(here, "numbers_flat.json")))
mp = os.path.join(here, "NUMBER_MAP.md")

# ---- the same stripping as check_numbers.py, so both tools see the same tokens
text = open(src).read()

# keep the abstract in scope: pull it out of the YAML block before the block is stripped
_fm = re.match(r"^---.*?---", text, flags=re.S)
_abs = re.search(r"(?ms)^abstract: \|\n((?:  .*\n?)+)", _fm.group(0)) if _fm else None
_abstract_text = _abs.group(1) if _abs else ""
text = re.sub(r"^---.*?---", "", text, count=1, flags=re.S)
text = _abstract_text + "\n" + text
text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
text = re.sub(r"`[^`]*`", " ", text)
text = re.sub(r"\[@[^\]]*\]", " ", text)
text = re.sub(r"https?://\S+", " ", text)
text = re.sub(r"\[YOU:[^\]]*\]", " ", text)
text = re.sub(r"(?m)^#+ .*$", " ", text)
text = re.sub(r"(?i)\b(fig(ure)?|table|section|sec|§|exp(eriment)?|seed|step)s?\.?\s*\d+(\.\d+)*", " ", text)
text = re.sub(r"\{#[^}]*\}", " ", text)
text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)  # ISO dates
TOKEN = re.compile(r"(?<![\w.])[-−+]?\d+(?:\.\d+)?(?:%|/\d+)?(?![\w])")
tokens = [t.replace("−", "-") for t in TOKEN.findall(text)]

# ---- the map
rows = {}
if os.path.exists(mp):
    for line in open(mp):
        if not line.startswith("|") or line.startswith("|--") or line.lower().startswith("| number"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        rows[cells[0].replace("−", "-")] = cells[1]


def value_of(key):
    if key.startswith("allow:"):
        return "allow"
    if key.startswith("derived:"):
        expr = key[len("derived:"):]
        # replace every ledger key in the expression by its value
        for k in sorted(flat, key=len, reverse=True):
            if k in expr:
                expr = expr.replace(k, repr(float(flat[k])))
        try:
            return float(eval(expr, {"__builtins__": {}}, {"abs": abs, "round": round}))
        except Exception as e:
            return f"ERR {e}"
    if key in flat:
        return float(flat[key])
    return None


def rounds_to(v, tok):
    if tok.endswith("%"):
        tok = tok[:-1]
    if "/" in tok:  # ratio like 5/6: numerator must equal the value
        tok = tok.split("/")[0]
    dec = len(tok.split(".")[1]) if "." in tok else 0
    x = float(tok)
    if dec == 0:
        return abs(v - x) < 0.5 + 1e-9 and (x == 0 or (v < 0) == (x < 0) or abs(v) < 0.5)
    return abs(round(v, dec) - x) < 1e-9


bad = []
seen = set()
for t in tokens:
    if t in seen:
        continue
    seen.add(t)
    if t.lstrip("-+").isdigit() and 1900 <= abs(float(t)) <= 2100:
        continue
    if t not in rows:
        bad.append(f"NO ROW      {t}")
        continue
    v = value_of(rows[t])
    if v == "allow":
        continue
    if v is None:
        bad.append(f"NO KEY      {t}  ->  {rows[t]}")
    elif isinstance(v, str):
        bad.append(f"BAD DERIVED {t}  ->  {rows[t]}  ({v})")
    elif not rounds_to(v, t):
        bad.append(f"MISMATCH    {t}  ->  {rows[t]} = {v}")

for b in bad:
    print(b)
if bad:
    print(f"audit_numbers: {len(bad)} problem(s) over {len(seen)} distinct tokens.")
    sys.exit(1)
print(f"audit_numbers: {len(seen)} distinct numeric tokens, every one mapped to a key with matching sign and rounding.")
