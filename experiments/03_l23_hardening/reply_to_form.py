#!/usr/bin/env python3
"""Turn a second rater's plain-text reply into RATINGS_FORM.md for form_to_json.py.

The rater gets photo_01..photo_28 (photo N is blind id h{N-1:02d} of the rater package)
and replies one line per photo, "N: copies broken fused warped pattern", e.g. "1: 0 1 0 2 0".
Those five words are the plain-language names of reduplication, fragmentation,
condensation, distortion and tiling, in that order (MESSAGE_for_rater.txt).

    python reply_to_form.py reply.txt RATINGS_FORM_<name>.md

Refuses to write a form unless all 28 photos are present exactly once with valid values.
"""
import re
import sys

FIELDS = ["reduplication", "fragmentation", "condensation", "distortion", "tiling"]
N_PHOTOS = 28


def parse(text: str) -> dict[int, list[int]]:
    rows, errors = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not re.match(r"^(?:photo\s*)?\d", line, flags=re.I):
            continue  # headers, the example line, blank lines
        m = re.match(r"^(?:photo\s*)?(\d{1,2})\s*[:.)-]?\s+(.*)$", line, flags=re.I)
        if not m:
            errors.append(f"cannot read: {raw!r}")
            continue
        n, rest = int(m.group(1)), m.group(2)
        vals = [int(v) for v in re.findall(r"\d+", rest)]
        if len(vals) != 5:
            errors.append(f"photo {n}: expected 5 numbers, got {len(vals)} in {raw!r}")
            continue
        if any(v > 3 for v in vals[:4]) or vals[4] not in (0, 1):
            errors.append(f"photo {n}: values out of range in {raw!r} (0-3, last one 0 or 1)")
            continue
        if n in rows:
            errors.append(f"photo {n}: appears twice")
        rows[n] = vals
    missing = [n for n in range(1, N_PHOTOS + 1) if n not in rows]
    extra = [n for n in rows if not 1 <= n <= N_PHOTOS]
    if missing:
        errors.append(f"missing photos: {missing}")
    if extra:
        errors.append(f"photo numbers outside 1-{N_PHOTOS}: {extra}")
    if errors:
        raise SystemExit("reply not converted:\n  " + "\n  ".join(errors))
    return rows


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = parse(open(src).read())
    out = ["# Ratings form (second rater)", "",
           "Converted from the rater's plain-text reply by reply_to_form.py; photo N = h{N-1}.", "",
           "| image | " + " | ".join(FIELDS) + " |", "|" + "---|" * (len(FIELDS) + 1)]
    for n in range(1, N_PHOTOS + 1):
        out.append(f"| h{n - 1:02d} | " + " | ".join(str(v) for v in rows[n]) + " |")
    open(dst, "w").write("\n".join(out) + "\n")
    print(f"{len(rows)} photos -> {dst}")


if __name__ == "__main__":
    main()
