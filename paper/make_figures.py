#!/usr/bin/env python3
"""Build the six figures for Preprint I from the authoritative JSONs.

    ../.venv/bin/python make_figures.py

Figure 1  living-room pair (the phenomenon), seed 44, g = 1 vs g = 11, both models
Figure 2  composite dose-response with the quality score on the same standardised axis
Figure 3  per-guidance means on a linear axis with the registered and log2-g slopes (fig3_slopes.png)
Figure 4  distortion vs guidance per prompt, raw and with veridicality adjustment (SDXL)
Figure 5  the painterly still life the author and Qwen3-VL-32B scored far apart (fig5_painterly_stilllife.png)
Figure 6  judge screen: fraction of non-zero scores per field on the 28 blind-subset images

Nothing here is a new analysis. Figure 3 draws slopes read from the reports. Figure 4 recomputes analysis_axes.md section 5.1 with
the same statlib.partial_spearman and asserts against the published table. Figure 6
counts non-zero scores in the probe files and the full-run judgements on the same 28
images the human rated.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "experiments" / "03_l23_hardening"
RES = E3 / "results-local"
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(E3))
import statlib as S  # noqa: E402

# palette: dataviz reference instance (references/palette.md), light surface
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
})
FIELDS = ["reduplication", "fragmentation", "condensation", "distortion"]
G = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 15.0]
MODEL_NAME = {"sdxl": "SDXL", "sd35": "SD 3.5"}
PROMPT_NAME = {"p1_stilllife": "still life", "p2_portrait": "portrait", "p3_bicycle": "bicycle",
               "p4_oranges": "oranges", "p5_livingroom": "living room", "p6_forest": "forest"}


def load(p):
    return json.load(open(p))


# ------------------------------------------------------------------ figure 1
def _composites():
    """Per-record z-composite from analyze.py, the same metric the report uses."""
    import analyze as A
    out = {}
    for m in ["sdxl", "sd35"]:
        cond, unc, notes = A._load_records(m, ["claude", "qwen"]); A._build_metric(cond, unc, notes)
        out[m] = cond
    return out


def fig1():
    """Living room, g = 1 and g = 11. With ten seeds there is no single median row:
    the central example is the upper of the two middle-ranked composites. The other
    g=1 column is the highest-scoring seed. Picks are recorded in fig1_data.json."""
    comps = _composites()
    picks = {}
    for m in ["sdxl", "sd35"]:
        rows = sorted((r["composite"], r["seed"]) for r in comps[m]
                      if r["prompt_id"] == "p5_livingroom" and r["guidance"] == 1.0)
        rows11 = sorted((r["composite"], r["seed"]) for r in comps[m]
                        if r["prompt_id"] == "p5_livingroom" and r["guidance"] == 11.0)
        picks[m] = {"upper_middle_g1": rows[len(rows) // 2][1], "top_g1": rows[-1][1],
                    "upper_middle_g11": rows11[len(rows11) // 2][1]}
    cols = [("upper_middle_g1", 1, "g = 1, upper-middle seed"),
            ("top_g1", 1, "g = 1, highest-scoring seed"),
            ("upper_middle_g11", 11, "g = 11, upper-middle seed")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1))
    for i, m in enumerate(["sdxl", "sd35"]):
        for j, (key, g, title) in enumerate(cols):
            seed = picks[m][key]
            im = Image.open(RES / m / f"p5_livingroom_g{g}_s{seed}.png").resize((512, 512))
            ax = axes[i, j]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0: ax.set_title(title, color=INK, fontsize=9)
            if j == 0: ax.set_ylabel(MODEL_NAME[m], color=INK, fontsize=10)
            ax.text(0.02, 0.02, f"seed {seed}", transform=ax.transAxes, fontsize=7, color="white",
                    bbox=dict(facecolor=INK, alpha=0.6, pad=1.5, edgecolor="none"))
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    fig.savefig(OUT / "fig1_livingroom_pair.png"); plt.close(fig)
    (OUT / "fig1_data.json").write_text(json.dumps({"note": "seeds shown in Figure 1, chosen by the per-image z-composite (mean of both judges) on the living-room prompt", "seeds": picks}, indent=1))


# ------------------------------------------------------------------ figure 2
def fig2():
    comps = _composites()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, m in zip(axes, ["sdxl", "sd35"]):
        r = load(RES / m / "l23_report_claude-qwen.json")
        comp = [r["composite_mean_by_guidance"][f"{g:.1f}"] for g in G]
        q = [r["quality_mean_by_guidance"][f"{g:.1f}"] for g in G]
        unc = r["baseline_composite_uncond"]
        ax.set_xscale("log")
        # Per-prompt traces (thin) and a prompt-cluster bootstrap 95% band.
        # Resample the six complete prompt clusters rather than treating images
        # as independent replicates.
        recs = comps[m]; rng = np.random.default_rng(0)
        for p in PROMPT_NAME:
            ys = [np.mean([r["composite"] for r in recs if r["prompt_id"] == p and r["guidance"] == g]) for g in G]
            ax.plot(G, ys, color=BLUE, lw=0.7, alpha=0.25, zorder=1)
        lo, hi = [], []
        for g in G:
            clusters = [
                np.array([r["composite"] for r in recs
                          if r["guidance"] == g and r["prompt_id"] == p])
                for p in PROMPT_NAME
            ]
            bs = [
                np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]).mean()
                for _ in range(2000)
            ]
            lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ax.fill_between(G, lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=1)
        ax.plot(G, comp, color=BLUE, lw=2, marker="o", ms=4.5, mec="white", mew=1, label="breakdown composite (mean, 95% band; thin lines: prompts)")
        ax.plot(G, q, color=ORANGE, lw=2, marker="s", ms=4, mec="white", mew=1, label="image quality")
        ax.plot([22], [unc], marker="o", ms=5.5, color=BLUE, mec="white", mew=1, ls="none")
        ax.annotate("empty\nprompt", (22, unc), xytext=(-7, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.5, color=INK2)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_xticks(G + [22]); ax.set_xticklabels([str(int(g)) for g in G] + ["∅"])
        ax.minorticks_off()
        ax.set_title(MODEL_NAME[m], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("classifier-free guidance g (log axis)")
        ax.set_ylim(-1.0, 2.4)
    axes[0].set_ylabel("mean score, standardised units")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig2_dose_response.png"); plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig3():
    m = "sdxl"
    kl = {j: load(RES / m / f"judgements_{j}.json")["images"] for j in ["claude", "qwen"]}
    ax_ = {j: load(RES / m / f"judgements_{j}_axes.json")["images"] for j in ["claude", "qwen"]}
    rows = []
    for p, name in PROMPT_NAME.items():
        ok = lambda d, k, f: k in d and f in d[k]  # error records ({"error": ...}) are skipped
        keys = sorted(k for k in kl["claude"] if k.startswith(p + "_g") and "uncond" not in k
                      and all(ok(kl[j], k, "distortion") for j in kl)
                      and all(ok(ax_[j], k, "veridicality") for j in ax_))
        g = np.array([float(k.split("_g")[1].split("_")[0]) for k in keys])
        dist = np.array([np.mean([kl[j][k]["distortion"] for j in kl]) for k in keys])
        ver = np.array([np.mean([ax_[j][k]["veridicality"] for j in ax_]) for k in keys])
        raw = S.spearman(dist, g); part = S.partial_spearman(dist, g, ver)
        rows.append((name, raw, part, len(keys)))
    published = {"oranges": (-0.702, -0.084), "bicycle": (-0.734, -0.261),
                 "still life": (-0.547, -0.216), "living room": (-0.530, -0.385),
                 "forest": (-0.503, -0.328), "portrait": (-0.231, 0.240)}
    for name, raw, part, n in rows:
        pr, pp = published[name]
        assert abs(raw - pr) < 0.006 and abs(part - pp) < 0.006, (name, raw, part, pr, pp)
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    y = np.arange(len(rows))
    for yi, (name, raw, part, n) in zip(y, rows):
        ax.plot([raw, part], [yi, yi], color=AXIS, lw=1.6, zorder=1)
    ax.scatter([r[1] for r in rows], y, color=BLUE, s=42, zorder=3, ec="white", lw=1, label="raw")
    ax.scatter([r[2] for r in rows], y, color=ORANGE, s=42, zorder=3, ec="white", lw=1, marker="s",
               label="adjusted for veridicality")
    ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel("Spearman correlation, distortion vs guidance (SDXL)")
    ax.set_xlim(-0.85, 0.4); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_style_confound.png"); plt.close(fig)
    (OUT / "fig3_data.json").write_text(json.dumps({
        "note": "SDXL, judge-mean distortion vs guidance per prompt; partial rank correlation adjusts for judge-mean veridicality (analysis_axes.md 5.1)",
        "per_prompt": {name: {"raw_spearman": round(raw, 4), "partial_controlling_veridicality": round(part, 4), "n": n}
                       for name, raw, part, n in rows}}, indent=1))
    return rows


# ------------------------------------------------------------------ figure 4
def fig4():
    subset = load(E3 / "human_subset.json")["items"]
    fields = FIELDS + ["tiling"]
    raters = []  # (label, {blind_id: record})
    probes = [("Qwen2.5-VL-7B", "probe_Qwen2.5-VL-7B-Instruct-4bit.json"),
              ("Qwen3-VL-8B", "probe_Qwen3-VL-8B-Instruct-4bit.json"),
              ("Gemma-3-27B", "probe_gemma-3-27b-it-qat-4bit.json"),
              ("Qwen3-VL-32B", "probe_Qwen3-VL-32B-Instruct-4bit.json")]
    for label, f in probes:
        raters.append((label, load(E3 / "probes" / f)["records"]))
    # full-run judges on the same 28 images
    full = {}
    for j, label in [("llama", "Llama-3.2-11B"), ("claude", "Claude Sonnet 5")]:
        recs = {}
        for it in subset:
            d = load(RES / it["model"] / f"judgements_{j}.json")["images"]
            recs[it["blind_id"]] = d[it["filename"]]
        full[label] = recs
    raters.insert(2, ("Llama-3.2-11B", full["Llama-3.2-11B"]))
    raters.append(("Claude Sonnet 5", full["Claude Sonnet 5"]))
    raters.append(("Human (author)", load(E3 / "human_ratings.json")))
    # a judge reply that failed to parse is not a zero; it is dropped from that rater's denominator
    M = np.array([[np.mean([recs[b][f] > 0 for b in recs if f in recs[b]]) for f in fields]
                  for _, recs in raters])
    MEAN = np.array([[np.mean([recs[b][f] for b in recs if f in recs[b]]) for f in fields]
                     for _, recs in raters])
    n_valid = {label: sum(1 for b in recs if "distortion" in recs[b]) for label, recs in raters}
    print("  valid replies on the 28:", n_valid)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(fields))); ax.set_xticklabels(fields, color=INK, fontsize=8.5)
    ax.set_yticks(range(len(raters))); ax.set_yticklabels([r[0] for r in raters], color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v*100:.0f}%\n{MEAN[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else INK, linespacing=1.1)
    ax.set_xlabel("per cell: percent of 28 images scored above zero (top); mean score, 0 to 3 (bottom)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_judge_screen.png"); plt.close(fig)
    (OUT / "fig4_data.json").write_text(json.dumps({
        "note": "fraction of the 28 blind-subset images each rater scored above zero, per field; parse errors excluded from the denominator",
        "n_valid": n_valid,
        "fraction_nonzero": {label: {f: round(float(v), 4) for f, v in zip(fields, row)} for (label, _), row in zip(raters, M)},
        "mean_score": {label: {f: round(float(v), 4) for f, v in zip(fields, row)} for (label, _), row in zip(raters, MEAN)}},
        indent=1))
    return raters, fields, M


# ------------------------------------------------------------------ figure 3 (slopes)
def fig_slopes():
    """Per-guidance mean composite on a linear g axis, with the registered straight-line
    LMM slope and the log2-g sensitivity slope drawn through the overall mean. Slopes are
    read from the reports (asserted), not refitted; lines are drawn in g units by undoing
    the same z-scoring analyze._lmm_slope applies."""
    import analyze as A
    comps = _composites()
    sens = load(E3 / "posthoc_sensitivity.json")["models"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    gg = np.linspace(1, 15, 200)
    out = {}
    for ax, m in zip(axes, ["sdxl", "sd35"]):
        recs = comps[m]
        g = np.array([r["guidance"] for r in recs], float)
        c = np.array([r["composite"] for r in recs], float)
        rep = load(RES / m / "l23_report_claude-qwen.json")
        b_lin = rep["primary_lmm"]["slope_standardized"]
        b_log = sens[m]["S1_covariate_scale"]["log2_g"]["slope_standardized"]
        mean_c = c.mean()
        lin = mean_c + b_lin * (gg - g.mean()) / g.std()
        lg = np.log2(g)
        logc = mean_c + b_log * (np.log2(gg) - lg.mean()) / lg.std()
        assert np.allclose(A._z(g), (g - g.mean()) / g.std())
        means = [c[g == x].mean() for x in G]
        ax.plot(gg, lin, color=BLUE, lw=2, label="straight-line slope (registered)", zorder=2)
        ax.plot(gg, logc, color=INK2, lw=1.4, ls=(0, (4, 3)), label="slope on log2 g (sensitivity)", zorder=2)
        ax.scatter(G, means, s=30, color=INK, zorder=3, label="mean breakdown at each g")
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_xticks(G); ax.set_xticklabels([str(int(x)) for x in G]); ax.minorticks_off()
        ax.set_xlabel("classifier-free guidance g (linear axis)")
        ax.set_title(f"{MODEL_NAME[m]}: slope {b_lin:.3f}; on log2 g {b_log:.3f}", color=INK, fontsize=9.5, loc="left")
        out[m] = {"slope_linear": round(b_lin, 4), "slope_log2": round(b_log, 4),
                  "means": {str(int(x)): round(float(v), 4) for x, v in zip(G, means)}}
    axes[0].set_ylabel("mean breakdown, standardised units")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig3_slopes.png"); plt.close(fig)
    (OUT / "fig3_slopes_data.json").write_text(json.dumps({"note": "Figure 3: slopes drawn from the reports; means per guidance value", "models": out}, indent=1))


# ------------------------------------------------------------------ figure 5 (painterly still life)
def fig_painterly():
    """The SDXL still life at g = 1, seed 50: the human-subset image the author and
    Qwen3-VL-32B scored far apart. Scores are read from the rating files and written out."""
    fname = "p1_stilllife_g1_s50.png"
    im = Image.open(RES / "sdxl" / fname).convert("RGB").resize((640, 640), Image.LANCZOS)
    im.save(OUT / "fig5_painterly_stilllife.png", optimize=True)
    subset = load(E3 / "human_subset.json")["items"]
    bid = next(it["blind_id"] for it in subset if it["model"] == "sdxl" and it["filename"] == fname)
    hum = load(E3 / "human_ratings.json")[bid]
    scores = {"human (author)": {f: hum[f] for f in FIELDS}}
    for j, label in [("qwen", "Qwen3-VL-32B"), ("claude", "Claude Sonnet 5")]:
        d = load(RES / "sdxl" / f"judgements_{j}.json")["images"][fname]
        scores[label] = {f: d[f] for f in FIELDS}
    (OUT / "fig5_painterly_data.json").write_text(json.dumps(
        {"note": "SDXL still life, g = 1, seed 50; human-subset blind id " + bid, "scores": scores}, indent=1))
    return scores


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig_slopes(); print("fig slopes ok")
    rows = fig3(); print("fig3 ok", [(r[0], round(r[1], 3), round(r[2], 3)) for r in rows])
    raters, fields, M = fig4(); print("fig4 ok")
    print("fig painterly", fig_painterly())
    for (label, _), row in zip(raters, M):
        print(f"  {label:16s}", " ".join(f"{v:.2f}" for v in row))
