#!/usr/bin/env python3
"""Post-process pandoc's paper.tex so no table splits across a page.

Every table in the paper is shorter than a page. Pandoc emits them as longtables, which
LaTeX may break between rows; ending each body row with \\\\* (longtable's no-break row
end) makes a table move whole to the next page instead. Usage: fix_tables.py paper.tex
"""
import re, sys

path = sys.argv[1]
tex = open(path).read()

def unbreak(m):
    body = m.group(0)
    head, sep, rows = body.partition("\\endlastfoot")
    if not sep:  # tables without a foot: protect everything after the first head
        head, sep, rows = body.partition("\\endhead")
    rows = re.sub(r"\\\\(\s*\n)", r"\\\\*\1", rows)
    return head + sep + rows

new, n = re.subn(r"\\begin\{longtable\}.*?\\end\{longtable\}", unbreak, tex, flags=re.S)
open(path, "w").write(new)
print(f"fix_tables: {n} tables made unbreakable")
