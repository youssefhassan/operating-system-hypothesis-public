#!/usr/bin/env python3
"""Convert a filled RATINGS_FORM.md into human_ratings_<rater>.json. Usage: form_to_json.py FORM.md NAME"""
import json, re, sys
rows = {}
for line in open(sys.argv[1]):
    m = re.match(r"\|\s*(h\d\d)\s*\|(.*)\|\s*$", line)
    if not m: continue
    cells = [c.strip() for c in m.group(2).split("|")]
    if len(cells) != 5 or not all(c.isdigit() for c in cells): continue
    v = list(map(int, cells))
    rows[m.group(1)] = dict(zip(["reduplication", "fragmentation", "condensation", "distortion", "tiling"], v))
out = f"human_ratings_{sys.argv[2]}.json"
json.dump(rows, open(out, "w"), indent=2); print(f"{len(rows)} rows -> {out}")
