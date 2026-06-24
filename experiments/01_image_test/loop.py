"""
Experiment 01 — agentic experiment loop (methodology-compliant).

Closed loop: generate (sweep_local) -> judge (judge.py, blind) -> aggregate
M(g) -> controller picks the next action from a RESTRICTED set -> stop on the
pre-registered rules in preregistration.json.

The objective is to CHARACTERIZE the effect and stop, never to "iterate until a
finding appears." The controller may only:
  - add_seeds:      raise N at an existing guidance value (shrink variance)
  - refine_grid:    insert a guidance value at a detected transition (within bounds)
  - replicate_model: run the next pre-registered architecture
  - stop:           confirm / null / budget

Changing the prompt, metric, rubric, thresholds, or grid bounds is NOT a loop
action; doing so reclassifies the run as exploratory (methodology Section 4) and
requires a dated commit.

Stats use numpy only (no scipy): Spearman via Pearson-on-ranks, a permutation
p-value, bootstrap CIs, and Cliff's delta for the low-vs-high effect.

Usage:
    python loop.py --dry-run            # plan only, no generation/judging
    python loop.py --init-seeds 3 --max-iters 8
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"
RESULTS_ROOT = HERE / "results-local"
AUDIT = RESULTS_ROOT / "iterations.jsonl"

FNAME_RE = re.compile(r"^g(?P<g>[0-9.]+)_s(?P<s>\d+)\.png$")
UNCOND_RE = re.compile(r"^uncond_s(?P<s>\d+)\.png$")


# ----------------------------- stats helpers -----------------------------------

def _ranks(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def spearman_perm_p(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    obs = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    yc = y.copy()
    hits = 0
    for _ in range(n):
        rng.shuffle(yc)
        if abs(spearman(x, yc)) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return float((gt - lt) / (len(a) * len(b)))


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        sa = rng.choice(a, len(a), replace=True)
        sb = rng.choice(b, len(b), replace=True)
        deltas.append(cliffs_delta(sa, sb))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ----------------------------- data plumbing -----------------------------------

def scan(model: str) -> dict:
    """Return {guidance: {seed: m_img}} and {'uncond': {seed: m_img}} for a model."""
    d = RESULTS_ROOT / model
    jpath = d / "judgements.json"
    judged = json.loads(jpath.read_text())["images"] if jpath.exists() else {}
    cond: dict[float, dict[int, float]] = {}
    uncond: dict[int, float] = {}
    for p in d.glob("*.png"):
        rec = judged.get(p.name, {})
        m = rec.get("m_img")
        mc = FNAME_RE.match(p.name)
        if mc:
            g, s = float(mc["g"]), int(mc["s"])
            cond.setdefault(g, {})[s] = m
            continue
        mu = UNCOND_RE.match(p.name)
        if mu:
            uncond[int(mu["s"])] = m
    return {"cond": cond, "uncond": uncond}


def aggregate(model: str, prereg: dict) -> dict:
    data = scan(model)["cond"]
    gs, means, ns, all_g, all_m = [], [], [], [], []
    for g in sorted(data):
        vals = [v for v in data[g].values() if v is not None]
        if not vals:
            continue
        gs.append(g)
        means.append(float(np.mean(vals)))
        ns.append(len(vals))
        all_g += [g] * len(vals)
        all_m += vals
    if len(gs) < 3:
        return {"model": model, "ready": False, "guidance": gs, "means": means, "n": ns}

    ag, am = np.array(all_g), np.array(all_m)
    rho = spearman(ag, am)
    p = spearman_perm_p(ag, am)
    low = am[ag <= prereg["bins"]["low_bin_guidance_max"]]
    high = am[ag >= prereg["bins"]["high_bin_guidance_min"]]
    delta = cliffs_delta(low, high)
    ci = bootstrap_delta_ci(low, high) if len(low) and len(high) else (0.0, 0.0)
    return {
        "model": model,
        "ready": True,
        "guidance": gs,
        "means": [round(m, 4) for m in means],
        "n": ns,
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p, 4),
        "low_vs_high_cliffs_delta": round(delta, 4),
        "delta_ci95": [round(ci[0], 4), round(ci[1], 4)],
    }


def verdict(agg: dict, prereg: dict) -> str:
    """confirm-eligible / null-eligible / inconclusive for a single model at full N."""
    if not agg.get("ready"):
        return "inconclusive"
    c = prereg["confirm"]
    ci = agg["delta_ci95"]
    confirm = (
        agg["spearman_rho"] <= c["spearman_rho_max"]
        and agg["spearman_p"] <= c["spearman_p_max"]
        and agg["low_vs_high_cliffs_delta"] >= c["low_minus_high_cliffs_delta_min"]
        and ci[0] > 0
    )
    if confirm:
        return "confirm-eligible"
    n = prereg["null"]
    if abs(agg["spearman_rho"]) <= n["spearman_abs_max"] and ci[0] <= 0 <= ci[1]:
        return "null-eligible"
    return "inconclusive"


# ----------------------------- the loop ----------------------------------------

def _seeds(n: int, start: int = 42) -> list[int]:
    return list(range(start, start + n))


def budget_used(prereg: dict) -> dict:
    imgs = 0
    for m in prereg["model_matrix"]["confirmatory"] + prereg["model_matrix"]["contrast_only_exploratory"]:
        d = RESULTS_ROOT / m
        if d.exists():
            imgs += len(list(d.glob("*.png")))
    return {"images": imgs, "judge_calls": imgs * len(prereg["judges"])}


def gen(model: str, guidance: list[float], seeds: list[int], uncond: bool, dry: bool) -> None:
    cmd = [sys.executable, "sweep_local.py", "--model", model,
           "--guidance", *[str(g) for g in guidance],
           "--seeds", *[str(s) for s in seeds]]
    if uncond:
        cmd.append("--unconditional")
    print(f"[loop] GEN: {' '.join(cmd)}")
    if not dry:
        subprocess.run(cmd, cwd=HERE, check=True)


def judge(model: str, prereg: dict, dry: bool) -> None:
    judges = prereg["judges"]
    flag = "both" if len(judges) > 1 else judges[0]
    cmd = [sys.executable, "judge.py", "--dir", f"results-local/{model}", "--judges", flag]
    print(f"[loop] JUDGE: {' '.join(cmd)}")
    if not dry:
        subprocess.run(cmd, cwd=HERE, check=True)


def controller(prereg: dict, init_seeds: int) -> dict:
    """Decide the next action from the restricted set. Pure function of disk state."""
    N = prereg["samples_per_setting"]
    grid = prereg["guidance_grid"]
    matrix = prereg["model_matrix"]["confirmatory"]
    bud = budget_used(prereg)
    if bud["images"] >= prereg["budget"]["max_images"] or bud["judge_calls"] >= prereg["budget"]["max_judge_calls"]:
        return {"action": "stop", "reason": "budget", "budget": bud}

    confirmed = []
    for model in matrix:
        data = scan(model)["cond"]
        # 1) nothing yet for this model -> seed the full coarse grid cheaply
        if not data:
            return {"action": "generate", "model": model,
                    "guidance": grid, "seeds": _seeds(init_seeds),
                    "uncond": True, "why": "initial coarse sweep"}
        # 2) under-sampled guidance values -> add seeds toward N
        for g in grid:
            have = len([v for v in data.get(g, {}).values() if v is not None])
            if have < N:
                start = 42 + have
                return {"action": "generate", "model": model,
                        "guidance": [g], "seeds": _seeds(N - have, start),
                        "uncond": False, "why": f"raise N at g={g} ({have}->{N})"}
        # 3) full N -> evaluate
        agg = aggregate(model, prereg)
        v = verdict(agg, prereg)
        if v == "confirm-eligible":
            confirmed.append(model)
        elif v == "null-eligible":
            return {"action": "stop", "reason": "null", "model": model, "agg": agg}
        # 4) inconclusive at full N -> refine grid once around the steepest gap
        else:
            gs = agg["guidance"]
            means = agg["means"]
            gaps = [(abs(means[i + 1] - means[i]), gs[i], gs[i + 1]) for i in range(len(gs) - 1)]
            gaps.sort(reverse=True)
            if gaps:
                _, lo, hi = gaps[0]
                mid = round((lo + hi) / 2, 2)
                if mid not in grid and grid[0] <= mid <= grid[-1] and mid not in data:
                    return {"action": "generate", "model": model,
                            "guidance": [mid], "seeds": _seeds(N),
                            "uncond": False, "why": f"refine grid at transition ~{mid}"}
            return {"action": "stop", "reason": "inconclusive", "model": model, "agg": agg}

    if len(confirmed) >= prereg["confirm"]["replicated_architectures_min"]:
        return {"action": "stop", "reason": "confirm", "models": confirmed}
    return {"action": "stop", "reason": "confirm-partial", "models": confirmed}


def log_iter(entry: dict) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    with AUDIT.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main(args: argparse.Namespace) -> None:
    prereg = json.loads(PREREG.read_text())
    print(f"[loop] hypothesis: {prereg['hypothesis'][:90]}...")
    for it in range(1, args.max_iters + 1):
        plan = controller(prereg, args.init_seeds)
        plan["iteration"] = it
        print(f"\n[loop] iter {it}: {plan.get('action')} — "
              f"{plan.get('why') or plan.get('reason')}")
        log_iter({"event": "plan", **{k: v for k, v in plan.items() if k != 'agg'}})
        if plan["action"] == "stop":
            print(f"[loop] STOP ({plan['reason']}). {json.dumps(plan.get('agg', plan.get('models', plan.get('budget', {}))), default=str)}")
            log_iter({"event": "stop", **plan})
            return
        gen(plan["model"], plan["guidance"], plan["seeds"], plan["uncond"], args.dry_run)
        judge(plan["model"], prereg, args.dry_run)
        agg = aggregate(plan["model"], prereg)
        print(f"[loop] {plan['model']} -> {json.dumps({k: agg.get(k) for k in ('spearman_rho','spearman_p','low_vs_high_cliffs_delta','delta_ci95','n')})}")
        log_iter({"event": "aggregate", "iteration": it, **agg})
        if args.dry_run:
            print("[loop] dry-run: stopping after one planned step.")
            return
        time.sleep(0.2)
    print("[loop] reached max-iters without a stop verdict.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 01 methodology-compliant agentic loop.")
    p.add_argument("--init-seeds", type=int, default=3, help="cheap coarse-sweep N before escalation")
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--dry-run", action="store_true", help="plan + print only; no generation/judging")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
