#!/usr/bin/env python3
"""Overclaim and voice scan for paper.md. Exits non-zero on any hit."""
import re, sys, os
src = sys.argv[1] if len(sys.argv) > 1 else 'paper.md'
text = open(src).read()
lines = text.split('\n')
hits = []
# From docs/REVIEW_program_I_2026-09-02.md section B: do not defend these.
patterns = [
    (r'analogy holds', 'review B: never as a result'),
    (r'form constants? (were|was|are|is) (found|detected|observed|present)', 'review B: Level-1 is a null'),
    (r'Level[- ]?[123]/?[23]?\b', 'review A7: project numbering, not Kluver\'s'),
    (r'precision relaxation', 'review A1: not settled; state the mapping as a modelling choice'),
    (r'\bproves?\b|\bdemonstrates? that the (brain|analogy)', 'overclaim'),
    (r'three judges', 'handover: the amended panel is two judges; two of three were inert'),
    (r'both (models|architectures) (tile|produce tilings)', 'review D2: SD 3.5 tiling rate is 0.00 on every prompt'),
    (r'most dissolved', 'handover 09-04: no per-image ranking exists'),
    (r'SD ?3\.5[^.]{0,80}\btil(es|ing)\b(?![^.]*(0\.00|never|not|no ))', 'review D2: SD 3.5 does not tile; say so or drop'),
    (r'Bredenberg[^.]*(rubric|axes|veridicality)|(rubric|axes|veridicality)[^.]*Bredenberg', 'the second rubric is Suzuki 2024; Bredenberg is motivation only'),
    (r'confirm(s|ed) on both', 'NUMBERS.md: the joint claim did not confirm'),
    (r'we (show|demonstrate) that the analogy', 'review B: nothing tests the analogy'),
    (r'\bwould have confirmed\b', 'analysis_sensitivity.md: do not say SD 3.5 would have confirmed'),
    (r'(pooled|overall) (rho|correlation|ρ)[^.]*[-−]0\.\d(?![^.]*(range|per.prompt|prompt.by.prompt|\bto\b))', 'methodology 5.2: a pooled number travels with its per-prompt range in the same sentence'),
    (r'—', 'voice: no em dashes'),
    (r'[“”‘’]', 'voice: no curly quotes (LaTeX handles quotes)'),
]
# Sentence-scoped patterns: negation may sit on an earlier line of a wrapped sentence, so
# these run over paragraphs joined into single lines and split at sentence ends.
sentence_patterns = [
    (r'(?<!not )(?<!no )(?<!nothing )\btransfers?\b.*(brain|biolog|operation)(?![^.]*(not claim|no biological|does not))', 'review B: no biological arm exists (negated uses are allowed)'),
    (r'^(?![^.]*(not call|not a null|reserved for|not read))(?=.*SD ?3\.5)(?=.*\bnull\b)', 'review A4: use SESOI / smaller-than-pre-specified language (negated uses are allowed)'),
]
para_start = 1
buf = []
def _flush(start):
    joined = ' '.join(x.strip() for x in buf)
    for sentence in re.split(r'(?<=[.!?])\s+', joined):
        for pat, why in sentence_patterns:
            if re.search(pat, sentence, flags=re.I):
                hits.append((start, why, sentence[:100]))
# verbatim code blocks (judge prompts, commands) are quoted text, not prose claims
_fence = False
_code = set()
for i, line in enumerate(lines, 1):
    if line.strip().startswith('```'):
        _fence = not _fence; _code.add(i); continue
    if _fence: _code.add(i)
for i, line in enumerate(lines, 1):
    if i in _code: continue
    if line.strip() == '' :
        if buf: _flush(para_start)
        buf = []; para_start = i + 1
    elif not line.strip().startswith('<!--'):
        buf.append(line)
if buf: _flush(para_start)
for i, line in enumerate(lines, 1):
    if line.strip().startswith('<!--') or i in _code: continue
    for pat, why in patterns:
        if re.search(pat, line, flags=re.I):
            hits.append((i, why, line.strip()[:100]))
for i, why, l in hits:
    print(f'{src}:{i}: [{why}] {l}')
if hits:
    sys.exit(1)
print('check_claims: clean.')
