#!/usr/bin/env python3
"""Fail if paper.md contains a number that is not in numbers_flat.json (at the precision
written) or in numbers_allow.txt. Usage: check_numbers.py paper.md NUMBERS_LEDGER.md"""
import json, re, sys, os
here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'paper.md')
flat = json.load(open(os.path.join(here, 'numbers_flat.json')))
vals = [float(v) for v in flat.values()]
allow = set()
ap = os.path.join(here, 'numbers_allow.txt')
if os.path.exists(ap):
    for line in open(ap):
        line = line.split('#')[0].strip()
        if line: allow.add(line)

text = open(src).read()
# strip YAML header, citations, urls, code, and inline latex labels

# keep the abstract in scope: pull it out of the YAML block before the block is stripped
_fm = re.match(r'^---.*?---', text, flags=re.S)
_abs = re.search(r'(?ms)^abstract: \|\n((?:  .*\n?)+)', _fm.group(0)) if _fm else None
_abstract_text = _abs.group(1) if _abs else ''
text = re.sub(r'^---.*?---', '', text, count=1, flags=re.S)
text = _abstract_text + '\n' + text
text = re.sub(r'`[^`]*`', ' ', text)
text = re.sub(r'\[@[^\]]*\]', ' ', text)
text = re.sub(r'https?://\S+', ' ', text)
text = re.sub(r'\[YOU:[^\]]*\]', ' ', text)
text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)             # html comments
text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', text)              # ISO dates
text = re.sub(r'\{#[^}]*\}', ' ', text)                          # pandoc attributes, e.g. {#fig:x width=92%}
text = re.sub(r'(?m)^#+ .*$', ' ', text)            # headings (section numbers)
text = re.sub(r'(?i)\b(fig(ure)?|table|section|sec|§|exp(eriment)?|seed|step)s?\.?\s*\d+(\.\d+)*', ' ', text)
tokens = re.findall(r'(?<![\w.])[-−+]?\d+(?:\.\d+)?(?![\w])', text)

def matches(tok):
    t = tok.replace('−', '-')
    if t in allow or t.lstrip('-+') in allow: return True
    x = float(t)
    if x.is_integer() and 1900 <= abs(x) <= 2100: return True   # years
    dec = len(t.split('.')[1]) if '.' in t else 0
    for v in vals:
        if round(v, dec) == x or (dec == 0 and abs(v - x) < 0.5 and abs(v) >= 1):
            return True
        if dec > 0 and abs(round(abs(v), dec) - abs(x)) < 1e-9:   # sign flips (e.g. -spont)
            return True
    return False

bad = sorted({t for t in tokens if not matches(t)}, key=lambda s: float(s.replace('−','-')))
if bad:
    print('UNLEDGERED NUMBERS:', ', '.join(bad))
    print('Add a JSON source (build_ledger.py) or a justified line in numbers_allow.txt.')
    sys.exit(1)
print(f'check_numbers: {len(tokens)} numeric tokens, all ledgered.')
