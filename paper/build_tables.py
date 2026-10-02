#!/usr/bin/env python3
"""Write the paper's results tables as pandoc pipe tables from numbers_flat.json.

    python3 build_tables.py        -> tables/T1_gates.md ... tables/T4_exp02_control.md

Writers include a table by pasting the file's contents (or a raw include) and never retype
a number. Every cell traces to the key listed in the trailing comment of each file, so the
audit can map it. Re-run after build_ledger.py.
"""
import json, os

here = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(here, "numbers_flat.json")))
OUT = os.path.join(here, "tables")
os.makedirs(OUT, exist_ok=True)


def g(key):
    return F[key]


def f2(x):  # signed, 3 decimals
    return f"{x:+.3f}".replace("+", "")


def ci(k):
    return f"[{f2(g(k+'[0]'))}, {f2(g(k+'[1]'))}]"


rows_used = []


def use(*keys):
    rows_used.extend(keys)
    return [g(k) for k in keys]


def write(name, title, header, body, note):
    lines = [f"Table: {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    # the verdict sentence lives in prose once; tables carry caption and keys only (Grok review 2026-09-17)
    lines += ["", "<!-- note (not pasted): " + note + " -->", "<!-- keys: " + "; ".join(sorted(set(rows_used))) + " -->", ""]
    open(os.path.join(OUT, name), "w").write("\n".join(lines))
    rows_used.clear()
    print("wrote", name)


# ---------------------------------------------------------------- T1 gates
M = {"sdxl": "SDXL", "sd35": "SD 3.5"}
body = []
for m in ["sdxl", "sd35"]:
    r = f"exp03.{m}.report"
    sl, lo, hi = use(f"{r}.primary_lmm.slope_standardized", f"{r}.primary_lmm.ci95[0]", f"{r}.primary_lmm.ci95[1]")
    pr, plo, phi = use(f"{r}.deconfound.partial_spearman_controlling_Q", f"{r}.deconfound.ci95[0]", f"{r}.deconfound.ci95[1]")
    nn, npmt = use(f"{r}.per_prompt.n_negative", f"{r}.per_prompt.n_prompts")
    k, = use(f"{r}.reliability.composite_weighted_kappa")
    body.append([M[m], f"{f2(sl)} [{f2(lo)}, {f2(hi)}]", f"{f2(pr)} [{f2(plo)}, {f2(phi)}]", f"{nn}/{npmt}", f"{k:.3f}"])
thr = use("exp03.prereg.confirm.lmm_composite_slope_standardized_max",
          "exp03.prereg.confirm.quality_controlled_partial_rho_max",
          "exp03.prereg.confirm.inter_judge_composite_weighted_kappa_min")
body.insert(0, ["pre-registered threshold", f"$\\le$ {thr[0]:.2f}", f"$\\le$ {thr[1]:.2f}", "$\\ge$ 5/6", f"$\\ge$ {thr[2]:.1f}"])
write("T1_gates.md", "Pre-registered gates, amended panel (Claude Sonnet 5 and Qwen3-VL-32B). Slope is the guidance-standardised fixed effect on the four-field composite from a linear mixed model with random prompt and seed intercepts, 95% CI. Partial rho adjusts for the two-proxy no-reference quality score. Prompts is the count with a negative per-prompt Spearman. Kappa is the quadratic-weighted composite between judges on all complete images, including ten empty-prompt baselines per checkpoint.",
      ["", "slope (95% CI)", "partial rho (95% CI)", "prompts negative", "kappa"], body,
      "The joint claim required both architectures to clear every gate. SDXL clears all four; SD 3.5 misses the slope gate and clears the other three.")

# ---------------------------------------------------------------- T2 sensitivity
body = []
cells = ["linear g (pre-registered)"]
for m in ["sdxl", "sd35"]:
    r = f"exp03.{m}.report.primary_lmm"
    sl, lo, hi = use(r + ".slope_standardized", r + ".ci95[0]", r + ".ci95[1]")
    cells.append(f"{f2(sl)} [{f2(lo)}, {f2(hi)}]")
body.append(cells)
for label, sub in [("log2 g", "S1_covariate_scale.log2_g"),
                   ("rank of g", "S1_covariate_scale.rank_g"),
                   ("three fields, no distortion, linear g", "S2_three_field_composite.lmm_slope_linear"),
                   ("three fields, no distortion, log2 g", "S2_three_field_composite.lmm_slope_log2_g")]:
    cells = [label]
    for m in ["sdxl", "sd35"]:
        k = f"exp03.sensitivity.models.{m}.{sub}"
        sl, lo, hi = use(k + ".slope_standardized", k + ".ci95[0]", k + ".ci95[1]")
        cells.append(f"{f2(sl)} [{f2(lo)}, {f2(hi)}]")
    body.append(cells)
cells = ["drop g = 1, Spearman composite vs g (post-hoc)"]
for m in ["sdxl", "sd35"]:
    k = f"exp03.posthoc.models.{m}.shape.excluding_g_le_1"
    rho, p = use(k + ".spearman_composite_vs_g", k + ".perm_p")
    cells.append(f"rho {f2(rho)}, p = {p:.4f}" if p >= 0.001 else f"rho {f2(rho)}, p < 0.001")
body.append(cells)
cells = ["share of the g = 1-to-curve-minimum drop occurring from g = 1 to 2 (post-hoc)"]
for m in ["sdxl", "sd35"]:
    s, = use(f"exp03.posthoc.models.{m}.shape.share_of_drop_in_first_step")
    cells.append(f"{100*s:.0f}%")
body.append(cells)
write("T2_sensitivity.md", "Sensitivity and post-hoc analyses of the guidance effect. Row 1 is the pre-registered primary slope. Rows 2 to 5 are mixed-model sensitivities pre-specified on 2026-09-03 with no gate. The last two rows are post-hoc shape descriptors from posthoc.py.",
      ["", "SDXL", "SD 3.5"], body,
      "None of these rows changes the pre-registered verdict. They locate the SD 3.5 miss: smaller than SDXL on every scale, below the pre-specified size of interest only on the pre-registered linear scale, and carried by g = 1.")

# ---------------------------------------------------------------- T3 axes
body = []
for m in ["sdxl", "sd35"]:
    r = f"exp03b.axes.models.{m}"
    sl, slo, shi = use(f"{r}.P1_veridicality_lmm.slope_standardized",
                       f"{r}.P1_veridicality_lmm.ci95[0]",
                       f"{r}.P1_veridicality_lmm.ci95[1]")
    pr, plo, phi = use(f"{r}.P2_dissociation.partial_spearman_controlling_kluver",
                       f"{r}.P2_dissociation.ci95[0]",
                       f"{r}.P2_dissociation.ci95[1]")
    k, = use(f"{r}.P3_reliability.composite_weighted_kappa")
    result = "all pass" if m == "sdxl" else "P3 only"
    body.append([M[m], f"{f2(sl)} [{f2(slo)}, {f2(shi)}]",
                 f"{f2(pr)} [{f2(plo)}, {f2(phi)}]", f"{k:.3f}", result])
write("T3_axes.md", "Pre-registered Exp 03b replication and dissociation gates on the Suzuki-adapted axes. P1 is the guidance-standardised veridicality LMM slope (registered pass >= +0.20, BH-adjusted p <= 0.05, CI above zero); P2 is partial Spearman of veridicality and guidance adjusted for the Klüver composite (pass >= +0.20, CI above zero); P3 is mean field-level quadratic-weighted kappa (pass >= 0.40). Result is the registered verdict.",
      ["", "P1 slope (95% CI)", "P2 partial rho (95% CI)", "P3 kappa", "result"], body,
      "The joint Exp 03b claim fails because SD 3.5 moves opposite to the registered P1 and P2 directions.")

# ---------------------------------------------------------------- T3 judges
body = []
for label, r in [("Claude + Qwen2.5-VL-7B (original judge B)", "report7b"),
                 ("Claude + Qwen2.5-VL-7B + Llama-3.2-11B (original three-judge panel)", "report3judge"),
                 ("Claude + Qwen3-VL-32B (amended panel)", "report")]:
    cells = [label]
    for m in ["sdxl", "sd35"]:
        k, n = use(f"exp03.{m}.{r}.reliability.composite_weighted_kappa",
                   f"exp03.{m}.{r}.reliability.n_images")
        cells.append(f"{k:.3f} (n = {int(n)})")
    body.append(cells)
hc, hlo, hhi = use("exp03.sdxl.report.human_reliability.human_vs_claude.composite_weighted_kappa",
                   "exp03.sdxl.report.human_reliability.human_vs_claude.composite_bootstrap.ci95[0]",
                   "exp03.sdxl.report.human_reliability.human_vs_claude.composite_bootstrap.ci95[1]")
hq, qlo, qhi = use("exp03.sdxl.report.human_reliability.human_vs_qwen.composite_weighted_kappa",
                   "exp03.sdxl.report.human_reliability.human_vs_qwen.composite_bootstrap.ci95[0]",
                   "exp03.sdxl.report.human_reliability.human_vs_qwen.composite_bootstrap.ci95[1]")
body.append(["human (author, n = 28) vs Claude Sonnet 5", f"{hc:.3f} [{hlo:.3f}, {hhi:.3f}]", "(pooled)"])
body.append(["human (author, n = 28) vs Qwen3-VL-32B", f"{hq:.3f} [{qlo:.3f}, {qhi:.3f}]", "(pooled)"])
r2c, r2clo, r2chi = use("exp03.rater2judges.rater2.human_vs_claude.composite_weighted_kappa",
                        "exp03.rater2judges.rater2.human_vs_claude.composite_bootstrap.ci95[0]",
                        "exp03.rater2judges.rater2.human_vs_claude.composite_bootstrap.ci95[1]")
r2q, r2qlo, r2qhi = use("exp03.rater2judges.rater2.human_vs_qwen.composite_weighted_kappa",
                        "exp03.rater2judges.rater2.human_vs_qwen.composite_bootstrap.ci95[0]",
                        "exp03.rater2judges.rater2.human_vs_qwen.composite_bootstrap.ci95[1]")
hh, hhlo, hhhi = use("exp03.humanhuman.composite_weighted_kappa",
                     "exp03.humanhuman.composite_bootstrap.ci95[0]", "exp03.humanhuman.composite_bootstrap.ci95[1]")
body.append(["second rater (naive, n = 28) vs Claude Sonnet 5", f"{r2c:.3f} [{r2clo:.3f}, {r2chi:.3f}]", "(pooled)"])
body.append(["second rater (naive, n = 28) vs Qwen3-VL-32B", f"{r2q:.3f} [{r2qlo:.3f}, {r2qhi:.3f}]", "(pooled)"])
body.append(["author vs second rater (human vs human, n = 28)", f"{hh:.3f} [{hhlo:.3f}, {hhhi:.3f}]", "(pooled)"])
write("T3_judges.md", "Composite (mean per-field) quadratic-weighted kappa across judge panels on the same generated corpus and rubric. Row-specific n is the complete-case image count and differs because Llama had missing replies. The first two rows are the panels as run on 2026-07-22; the third is the amended panel, after the Qwen2.5-VL-7B seat was refilled with Qwen3-VL-32B on 2026-08-08, with no rubric change. Human rows use a 28-image blind subset drawn from both checkpoints, too few to split, so each row has one pooled value, shown under SDXL; bootstrap 95% CI. The second-rater rows are exploratory, with no gate.",
      ["panel", "SDXL", "SD 3.5"], body,
      "Same family, same release, same quantisation; the kappa change is the judge, not the rubric.")

# ---------------------------------------------------------------- T4 exp02
body = []
for label, k in [("rendered form constants (mu >= 1.05 mu_c)", "positives"),
                 ("blank renders (mu <= 0.9 mu_c)", "blank_negatives"),
                 ("ordinary Exp 03 images (g in {7, 11})", "photo_negatives")]:
    kk, n, rate, lo, hi = use(f"exp02.report.positive_control.{k}.k", f"exp02.report.positive_control.{k}.n",
                              f"exp02.report.positive_control.{k}.rate", f"exp02.report.positive_control.{k}.ci95[0]",
                              f"exp02.report.positive_control.{k}.ci95[1]")
    body.append([label, f"{int(kk)}/{int(n)}", f"{rate*100:.0f}% [{max(lo,0.0)*100+0.0:.0f}%, {hi*100:.0f}%]"])
write("T4_exp02_control.md", "Exp 02 positive control for the Exp 01 form-constant judge (archived rubric, single blind call per image). A set counts as flagged when any of the four binary class flags fires. Pre-specified pass required the ordinary-image rate at or below 20%.",
      ["image set", "flagged", "rate (95% CI)"], body,
      "Sensitivity is perfect and the blank rate is zero; the ordinary-image rate fails the pre-specified ceiling. Every flagged ordinary image names a literal grid in its note.")

# ---------------------------------------------------------------- T5 per-guidance means (Figure 2 as numbers)
body = []
for gv in ["1.0", "2.0", "3.0", "5.0", "7.0", "11.0", "15.0"]:
    cells = [str(int(float(gv)))]
    for m in ["sdxl", "sd35"]:
        n, c, q = use(f"exp03.{m}.report.n_by_guidance.{gv}",
                      f"exp03.{m}.report.composite_mean_by_guidance.{gv}",
                      f"exp03.{m}.report.quality_mean_by_guidance.{gv}")
        cells += [str(int(n)), f2(c), f2(q)]
    body.append(cells)
cells = ["empty prompt"]
for m in ["sdxl", "sd35"]:
    u, n = use(f"exp03.{m}.report.baseline_composite_uncond",
               f"exp03.{m}.report.notes.n_uncond")
    cells += [str(int(n)), f2(u), "n/a"]
body.append(cells)
write("T5_dose_means.md", "The two lines of Figure 2 as numbers: complete-case n, mean breakdown composite and mean quality score Q at each guidance value, in standardised units within each checkpoint. Conditioned cells contain 58 to 60 images after parser failures; the empty-prompt row has no quality score in the pre-registered analysis.",
      ["g", "SDXL n", "SDXL composite", "SDXL Q", "SD 3.5 n", "SD 3.5 composite", "SD 3.5 Q"], body,
      "Figure 2 as a table, so it can be re-plotted without the repository.")
