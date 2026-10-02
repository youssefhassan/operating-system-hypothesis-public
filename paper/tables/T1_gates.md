Table: Pre-registered gates, amended panel (Claude Sonnet 5 and Qwen3-VL-32B). Slope is the guidance-standardised fixed effect on the four-field composite from a linear mixed model with random prompt and seed intercepts, 95% CI. Partial rho adjusts for the two-proxy no-reference quality score. Prompts is the count with a negative per-prompt Spearman. Kappa is the quadratic-weighted composite between judges on all complete images, including ten empty-prompt baselines per checkpoint.

|  | slope (95% CI) | partial rho (95% CI) | prompts negative | kappa |
|---|---|---|---|---|
| pre-registered threshold | $\le$ -0.20 | $\le$ -0.20 | $\ge$ 5/6 | $\ge$ 0.4 |
| SDXL | -0.340 [-0.400, -0.279] | -0.433 [-0.508, -0.347] | 6/6 | 0.562 |
| SD 3.5 | -0.182 [-0.248, -0.115] | -0.245 [-0.338, -0.146] | 5/6 | 0.440 |

<!-- note (not pasted): The joint claim required both architectures to clear every gate. SDXL clears all four; SD 3.5 misses the slope gate and clears the other three. -->
<!-- keys: exp03.prereg.confirm.inter_judge_composite_weighted_kappa_min; exp03.prereg.confirm.lmm_composite_slope_standardized_max; exp03.prereg.confirm.quality_controlled_partial_rho_max; exp03.sd35.report.deconfound.ci95[0]; exp03.sd35.report.deconfound.ci95[1]; exp03.sd35.report.deconfound.partial_spearman_controlling_Q; exp03.sd35.report.per_prompt.n_negative; exp03.sd35.report.per_prompt.n_prompts; exp03.sd35.report.primary_lmm.ci95[0]; exp03.sd35.report.primary_lmm.ci95[1]; exp03.sd35.report.primary_lmm.slope_standardized; exp03.sd35.report.reliability.composite_weighted_kappa; exp03.sdxl.report.deconfound.ci95[0]; exp03.sdxl.report.deconfound.ci95[1]; exp03.sdxl.report.deconfound.partial_spearman_controlling_Q; exp03.sdxl.report.per_prompt.n_negative; exp03.sdxl.report.per_prompt.n_prompts; exp03.sdxl.report.primary_lmm.ci95[0]; exp03.sdxl.report.primary_lmm.ci95[1]; exp03.sdxl.report.primary_lmm.slope_standardized; exp03.sdxl.report.reliability.composite_weighted_kappa -->
