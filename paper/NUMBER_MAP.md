# Number map: every number written in paper.md, and the ledger key it came from

Checked by `audit_numbers.py` (part of `make check`). One row per distinct number as
written. `key` is a `numbers_flat.json` key, or `allow:<reason>` for a design constant, or
`derived:<expression over ledger keys>` (evaluated). Writers append rows as they write; a
number with no row fails the build.

| number | key | note |
|---|---|---|
| 860 | allow:design, 430 images per model x 2 (exp03.prereg) | total images Exp 03 |
| 430 | allow:design, 6 prompts x 7 g x 10 seeds + 10 uncond | images per model Exp 03 |
| 200 | allow:design, Exp 01: 10 seeds x 9 g x 2 models + uncond | total images Exp 01 |
| 28 | exp03.sdxl.report.human_reliability.n_subset | human subset |
| 1.0 | allow:design, Exp 01 and Exp 03 guidance grid (exp01.prereg, exp03.prereg) | grid value |
| 1.5 | allow:design, Exp 01 guidance grid | grid value |
| 2.0 | allow:design, guidance grid | grid value |
| 3.0 | allow:design, guidance grid | grid value |
| 4.5 | allow:design, Exp 01 guidance grid | grid value |
| 5.0 | allow:design, Exp 03 guidance grid | grid value |
| 6.0 | allow:design, Exp 01 guidance grid | grid value |
| 7.0 | allow:design, Exp 03 guidance grid | grid value |
| 8.0 | allow:design, Exp 01 guidance grid | grid value |
| 11.0 | allow:design, guidance grid | grid value |
| 15.0 | allow:design, guidance grid | grid value |
| 9 | allow:design, Exp 01 has nine guidance values | count |
| 7 | allow:design, Exp 03 has seven guidance values | count |
| 6 | allow:design, six prompts | count |
| 10 | allow:design, ten seeds | count |
| 42 | allow:design, first seed | seed range |
| 51 | allow:design, last seed | seed range |
| 25 | allow:design, sampling steps | pinned parameter |
| 100 | allow:design, Exp 01 images per model | count |
| 1024 | allow:design, resolution | pinned parameter |
| 0 | allow:generic, rubric minimum / zero | generic |
| 1 | allow:generic, one prompt / binary flag / step numbers | generic |
| 2 | allow:generic, two rubrics, two judges, two architectures | generic |
| 3 | allow:generic, rubric maximum | generic |
| 4 | allow:generic, four fields, four steps | generic |
| 5 | allow:design, prompt-generality gate 5 of 6 (exp03.prereg.confirm.prompt_generality) | gate |
| 30 | allow:design, human subset target 25 to 30; judge size threshold about 30B | design |
| 3.5 | allow:name, SD 3.5 | model name |
| 3.2 | allow:name, Llama-3.2 | model name |
| 0.20 | allow:SESOI, exp03.prereg.confirm.lmm_composite_slope_standardized_max | pre-specified size of interest |
| +0.20 | allow:SESOI, Exp 03b P1/P2 registered pass floor of +0.20 | Exp 03b gate |
| -0.20 | exp03.prereg.confirm.lmm_composite_slope_standardized_max | gate: standardised slope and partial rho thresholds |
| 0.4 | exp03.prereg.confirm.inter_judge_composite_weighted_kappa_min | gate: inter-judge composite weighted kappa floor |
| 175 | allow:citation, Kluver 1942 chapter page range pp. 175-207 | page cite |
| 207 | allow:citation, Kluver 1942 chapter page range pp. 175-207 | page cite |
| 11 | allow:design, guidance grid value g = 11 | grid value |
| 15 | allow:design, guidance grid value g = 15 | grid value |
| 70 | fig3.data.per_prompt.bicycle.n | images per prompt, Exp 03b per-prompt axes |
| 5/6 | exp03.sd35.report.per_prompt.n_negative | SD 3.5 prompts with a negative per-prompt Spearman |
| 6/6 | exp03.sdxl.report.per_prompt.n_negative | SDXL prompts with a negative per-prompt Spearman |
| -0.340 | exp03.sdxl.report.primary_lmm.slope_standardized | SDXL pre-registered linear slope |
| -0.400 | exp03.sdxl.report.primary_lmm.ci95[0] | SDXL slope CI lower |
| -0.279 | exp03.sdxl.report.primary_lmm.ci95[1] | SDXL slope CI upper |
| -0.433 | exp03.sdxl.report.deconfound.partial_spearman_controlling_Q | SDXL quality-controlled partial rho |
| -0.508 | exp03.sdxl.report.deconfound.ci95[0] | SDXL partial rho CI lower |
| -0.347 | exp03.sdxl.report.deconfound.ci95[1] | SDXL partial rho CI upper |
| 0.562 | exp03.sdxl.report.reliability.composite_weighted_kappa | SDXL composite inter-judge kappa |
| -0.182 | exp03.sd35.report.primary_lmm.slope_standardized | SD 3.5 pre-registered linear slope |
| -0.248 | exp03.sd35.report.primary_lmm.ci95[0] | SD 3.5 slope CI lower (also three-field log2 CI upper) |
| -0.116 | exp03.sdxl.report.per_field_dose_response.reduplication.spearman_rho_vs_g | secondary family, SDXL reduplication |
| -0.115 | exp03.sd35.report.primary_lmm.ci95[1] | SD 3.5 slope CI upper (Powell fit) |
| -0.245 | exp03.sd35.report.deconfound.partial_spearman_controlling_Q | SD 3.5 quality-controlled partial rho |
| -0.338 | exp03.sd35.report.deconfound.ci95[0] | SD 3.5 partial rho CI lower |
| -0.146 | exp03.sd35.report.deconfound.ci95[1] | SD 3.5 partial rho CI upper |
| 0.440 | exp03.sd35.report.reliability.composite_weighted_kappa | SD 3.5 composite inter-judge kappa |
| -0.74 | exp03.sdxl.report.per_prompt.per_prompt_spearman_composite_vs_g.p3_bicycle | SDXL per-prompt range, low end |
| -0.28 | exp03.sdxl.report.per_prompt.per_prompt_spearman_composite_vs_g.p2_portrait | SDXL per-prompt range, high end (also SD 3.5 E3 CI lower) |
| -0.49 | exp03.sd35.report.per_prompt.per_prompt_spearman_composite_vs_g.p5_livingroom | SD 3.5 per-prompt range, low end |
| +0.30 | exp03.sd35.report.per_prompt.per_prompt_spearman_composite_vs_g.p2_portrait | SD 3.5 portrait, four-field composite |
| 0.49 | exp03.sdxl.report.matched_quality_arms.cliffs_delta_low_vs_high | SDXL matched-quality Cliff's delta |
| 0.362 | exp03.sdxl.report.matched_quality_arms.ci95[0] | SDXL Cliff's delta CI lower |
| 0.606 | exp03.sdxl.report.matched_quality_arms.ci95[1] | SDXL Cliff's delta CI upper |
| 0.188 | exp03.sd35.report.matched_quality_arms.cliffs_delta_low_vs_high | SD 3.5 matched-quality Cliff's delta |
| 0.050 | exp03.sd35.report.matched_quality_arms.ci95[0] | SD 3.5 Cliff's delta CI lower |
| 0.320 | exp03.sd35.report.matched_quality_arms.ci95[1] | SD 3.5 Cliff's delta CI upper |
| 1.51 | exp03.sdxl.report.baseline_composite_uncond | SDXL empty-prompt composite |
| 2.17 | exp03.sd35.report.baseline_composite_uncond | SD 3.5 empty-prompt composite |
| -0.440 | exp03.sensitivity.models.sdxl.S1_covariate_scale.log2_g.slope_standardized | sensitivity: SDXL log2 slope |
| -0.494 | exp03.sensitivity.models.sdxl.S1_covariate_scale.log2_g.ci95[0] | sensitivity CI |
| -0.386 | exp03.sensitivity.models.sdxl.S1_covariate_scale.log2_g.ci95[1] | sensitivity CI |
| -0.301 | exp03.sensitivity.models.sd35.S1_covariate_scale.log2_g.slope_standardized | sensitivity: SD 3.5 log2 slope |
| -0.362 | exp03.sensitivity.models.sd35.S1_covariate_scale.log2_g.ci95[0] | sensitivity CI |
| -0.239 | exp03.sensitivity.models.sd35.S1_covariate_scale.log2_g.ci95[1] | sensitivity CI |
| -0.415 | exp03.sensitivity.models.sdxl.S1_covariate_scale.rank_g.slope_standardized | sensitivity: SDXL rank slope |
| -0.471 | exp03.sensitivity.models.sdxl.S1_covariate_scale.rank_g.ci95[0] | sensitivity CI |
| -0.359 | exp03.sensitivity.models.sdxl.S1_covariate_scale.rank_g.ci95[1] | sensitivity CI |
| -0.265 | exp03.sensitivity.models.sd35.S1_covariate_scale.rank_g.slope_standardized | sensitivity: SD 3.5 rank slope |
| -0.328 | exp03.sensitivity.models.sd35.S1_covariate_scale.rank_g.ci95[0] | sensitivity CI |
| -0.202 | exp03.sensitivity.models.sd35.S1_covariate_scale.rank_g.ci95[1] | sensitivity CI |
| -0.308 | exp03.sensitivity.models.sdxl.S2_three_field_composite.lmm_slope_linear.slope_standardized | sensitivity: SDXL three-field linear slope |
| -0.365 | exp03.sensitivity.models.sdxl.S2_three_field_composite.lmm_slope_linear.ci95[0] | sensitivity CI |
| -0.251 | exp03.sensitivity.models.sdxl.S2_three_field_composite.lmm_slope_linear.ci95[1] | sensitivity CI |
| -0.206 | exp03.sensitivity.models.sd35.S2_three_field_composite.lmm_slope_linear.slope_standardized | sensitivity: SD 3.5 three-field linear slope |
| -0.269 | exp03.sensitivity.models.sd35.S2_three_field_composite.lmm_slope_linear.ci95[0] | sensitivity CI |
| -0.144 | exp03.sensitivity.models.sd35.S2_three_field_composite.lmm_slope_linear.ci95[1] | sensitivity CI |
| -0.395 | exp03.sensitivity.models.sdxl.S2_three_field_composite.lmm_slope_log2_g.slope_standardized | sensitivity: SDXL three-field log2 slope |
| -0.447 | exp03.sensitivity.models.sdxl.S2_three_field_composite.lmm_slope_log2_g.ci95[0] | sensitivity CI |
| -0.344 | exp03.sensitivity.models.sdxl.S2_three_field_composite.lmm_slope_log2_g.ci95[1] | sensitivity CI |
| -0.307 | exp03.sensitivity.models.sd35.S2_three_field_composite.lmm_slope_log2_g.slope_standardized | sensitivity: SD 3.5 three-field log2 slope |
| -0.42 | exp03.sensitivity.models.sd35.S2_three_field_composite.per_prompt.per_prompt_spearman_composite_vs_g.p2_portrait | sensitivity: SD 3.5 portrait, three-field |
| -0.244 | exp03.posthoc.models.sdxl.shape.excluding_g_le_1.spearman_composite_vs_g | post-hoc: SDXL Spearman, g = 1 dropped |
| 0.0002 | exp03.posthoc.models.sdxl.shape.excluding_g_le_1.perm_p | post-hoc: permutation p, SDXL |
| 0.001 | allow:p-value reporting floor in Table 2 for exp03.posthoc.models.sdxl.shape.excluding_g_le_1.perm_p = 0.0002 | threshold |
| 0.048 | exp03.posthoc.models.sd35.shape.excluding_g_le_1.spearman_composite_vs_g | post-hoc: SD 3.5 Spearman, g = 1 dropped (table form) |
| +0.048 | exp03.posthoc.models.sd35.shape.excluding_g_le_1.spearman_composite_vs_g | post-hoc: SD 3.5 Spearman, g = 1 dropped (prose form) |
| 0.3757 | exp03.posthoc.models.sd35.shape.excluding_g_le_1.perm_p | post-hoc: permutation p, SD 3.5 (table form) |
| 0.38 | exp03.posthoc.models.sd35.shape.excluding_g_le_1.perm_p | post-hoc: permutation p, SD 3.5 (prose form) |
| 60% | derived:100*exp03.posthoc.models.sdxl.shape.share_of_drop_in_first_step | post-hoc: share of drop in the first step, SDXL |
| 91% | derived:100*exp03.posthoc.models.sd35.shape.share_of_drop_in_first_step | post-hoc: share of drop in the first step, SD 3.5 |
| -0.486 | exp03b.axes.models.sdxl.exploratory.E3_painterly_confound.raw_spearman_distortion_vs_g | SDXL distortion vs g, raw |
| -0.175 | exp03b.axes.models.sdxl.exploratory.E3_painterly_confound.partial_controlling_veridicality | SDXL distortion vs g, veridicality held fixed |
| -0.27 | exp03b.axes.models.sdxl.exploratory.E3_painterly_confound.ci95[0] | SDXL partial CI lower |
| -0.08 | exp03b.axes.models.sdxl.exploratory.E3_painterly_confound.ci95[1] | SDXL partial CI upper (also oranges partial, SD 3.5 partial CI upper) |
| 64% | derived:100*exp03b.axes.models.sdxl.exploratory.E3_painterly_confound.proportional_attenuation_in_absolute_association | share as percent |
| -0.70 | fig3.data.per_prompt.oranges.raw_spearman | oranges, raw |
| -0.53 | fig3.data.per_prompt.living room.raw_spearman | living room, raw |
| -0.385 | fig3.data.per_prompt.living room.partial_controlling_veridicality | living room, veridicality held fixed |
| -0.23 | fig3.data.per_prompt.portrait.raw_spearman | portrait, raw |
| +0.24 | fig3.data.per_prompt.portrait.partial_controlling_veridicality | portrait, veridicality held fixed |
| -0.114 | exp03b.axes.models.sd35.exploratory.E3_painterly_confound.raw_spearman_distortion_vs_g | SD 3.5 distortion vs g, raw |
| -0.179 | exp03b.axes.models.sd35.exploratory.E3_painterly_confound.partial_controlling_veridicality | SD 3.5 distortion vs g, veridicality held fixed |
| 1.07 | exp03b.axes.models.sd35.exploratory.E1_overshoot.drop_from_peak | exploratory: SD 3.5 veridicality drop from peak |
| 0.16 | exp03b.axes.models.sdxl.exploratory.E1_overshoot.drop_from_peak | exploratory: SDXL veridicality drop from peak |
| 95% | allow:design, confidence level reported in the table captions | statistics |
| 4.6 | allow:name, Claude Sonnet 4.6 (the Exp 01 judge) | model name |
| +0.01 | exp01.sdxl.report.confirmatory.image_level_spearman_rho_descriptive | Exp 01: image-level M vs guidance, SDXL |
| +0.10 | exp01.sd35.report.confirmatory.image_level_spearman_rho_descriptive | Exp 01: image-level M vs guidance, SD 3.5 |
| -0.044 | exp01.sdxl.report.confirmatory.spearman_rho | Exp 01: registered M(g) vs guidance, SDXL |
| +0.234 | exp01.sd35.report.confirmatory.spearman_rho | Exp 01: registered M(g) vs guidance, SD 3.5 |
| 0.033 | exp01.sdxl.report.confirmatory.means[6] | Exp 01 lowest per-guidance mean M |
| 0.167 | exp01.sdxl.report.confirmatory.means[0] | Exp 01 highest per-guidance mean M |
| -0.6 | exp01.prereg.confirm.spearman_rho_max | Exp 01 pre-registered confirmation threshold |
| 0.3 | exp01.prereg.null.spearman_abs_max | Exp 01 pre-registered null region |
| 1.00 | exp02.report.gates.above_threshold_score.value | Exp 02: mean M at supra-threshold gain (also control rate on positives) |
| 0.00 | exp02.report.gates.blank_baseline_score.value | Exp 02: mean M on blank renders (also control rate on blanks) |
| 0.87 | exp02.report.trend.hex.spearman_rho | Exp 02: M vs gain, hexagon regime |
| 0.91 | exp02.report.trend.stripe.spearman_rho | Exp 02: M vs gain, stripe regime |
| 80 | exp02.report.positive_control.positives.n | Exp 02: rendered form constants in the control |
| 18 | exp02.report.positive_control.photo_negatives.k | Exp 02: flagged ordinary images |
| 40 | exp02.report.positive_control.photo_negatives.n | Exp 02: ordinary images in the control |
| 8 | allow:design, 8 portrait images among the 40 Exp 02 photographic negatives (02 analysis.md section 2) | count |
| 1.05 | exp02.prereg.mu_grid_in_units_of_mu_c[4] | Exp 02 gain grid: supra-threshold cut for the control |
| 0.9 | exp02.prereg.mu_grid_in_units_of_mu_c[1] | Exp 02 gain grid: blank cut for the control |
| 80/80 | exp02.report.positive_control.positives.k | Exp 02 control: flagged rendered form constants |
| 0.95 | exp02.report.positive_control.positives.ci95[0] | Exp 02 control: positives CI lower |
| 0/40 | exp02.report.positive_control.blank_negatives.k | Exp 02 control: flagged blank renders |
| -0.00 | exp02.report.positive_control.blank_negatives.ci95[0] | Exp 02 control: blanks CI lower |
| 0.09 | exp02.report.positive_control.blank_negatives.ci95[1] | Exp 02 control: blanks CI upper |
| 18/40 | exp02.report.positive_control.photo_negatives.k | Exp 02 control: flagged ordinary images |
| 0.45 | exp02.report.positive_control.photo_negatives.rate | Exp 02 control: ordinary-image flag rate |
| 0.31 | exp02.report.positive_control.photo_negatives.ci95[0] | Exp 02 control: ordinary-image CI lower |
| 0.60 | exp02.report.positive_control.photo_negatives.ci95[1] | Exp 02 control: ordinary-image CI upper |
| 0.290 | exp03.sdxl.report7b.reliability.composite_weighted_kappa | SDXL composite kappa, original panel with the 7B judge |
| 0.158 | exp03.sd35.report7b.reliability.composite_weighted_kappa | SD 3.5 composite kappa, original panel with the 7B judge |
| 0.126 | exp03.sdxl.report3judge.reliability.composite_weighted_kappa | SDXL composite kappa, original panel of three |
| 0.135 | exp03.sd35.report3judge.reliability.composite_weighted_kappa | SD 3.5 composite kappa, original panel of three |
| 0.337 | exp03.sdxl.report.human_reliability.human_vs_claude.composite_weighted_kappa | human vs Claude composite kappa |
| 0.106 | exp03.sdxl.report.human_reliability.human_vs_claude.composite_bootstrap.ci95[0] | human vs Claude CI lower |
| 0.495 | exp03.sdxl.report.human_reliability.human_vs_claude.composite_bootstrap.ci95[1] | human vs Claude CI upper |
| 0.124 | exp03.sdxl.report.human_reliability.human_vs_qwen.composite_weighted_kappa | human vs Qwen3-VL-32B composite kappa |
| -0.059 | exp03.sdxl.report.human_reliability.human_vs_qwen.composite_bootstrap.ci95[0] | human vs Qwen CI lower |
| 0.313 | exp03.sdxl.report.human_reliability.human_vs_qwen.composite_bootstrap.ci95[1] | human vs Qwen CI upper |
| 0.212 | exp03.sdxl.report.human_reliability.human_vs_claude.distortion | weakest field, human vs Claude |
| 0.18 | fig4.data.fraction_nonzero.Llama-3.2-11B.reduplication | judge screen: Llama non-zero share, reduplication |
| 0.36 | fig4.data.fraction_nonzero.Gemma-3-27B.reduplication | judge screen: Gemma non-zero share, reduplication |
| 0.29 | fig4.data.fraction_nonzero.Gemma-3-27B.distortion | judge screen: Gemma non-zero share, distortion |
| -0.618 | exp03.sdxl.report.per_field_dose_response.fragmentation.spearman_rho_vs_g | secondary family, SDXL fragmentation |
| -0.442 | exp03.sdxl.report.per_field_dose_response.condensation.spearman_rho_vs_g | secondary family, SDXL condensation |
| 0.018 | exp03.sdxl.report.per_field_dose_response.reduplication.bh_q | secondary family, largest BH q on SDXL |
| -0.453 | exp03.sd35.report.per_field_dose_response.fragmentation.spearman_rho_vs_g | secondary family, SD 3.5 fragmentation |
| 0.05 | allow:BH threshold / alpha (also numbers_allow.txt) | q-value bar for the secondary family |
| -0.62 | exp03.posthoc.models.sdxl.per_prompt_per_field.p6_forest.composite_rho | post-hoc, forest composite, SDXL |
| +0.05 | exp03.posthoc.models.sdxl.per_prompt_per_field.p6_forest.reduplication.rho | post-hoc, forest reduplication, SDXL |
| 0.40 | exp03.posthoc.models.sdxl.tiling_rate.p6_forest.at_lowest_g | post-hoc, forest tiling rate at lowest g, SDXL |
| -0.02 | exp03.posthoc.models.sd35.per_prompt_per_field.p6_forest.composite_rho | post-hoc, forest composite, SD 3.5 |
| +0.52 | exp03.posthoc.models.sd35.per_prompt_per_field.p6_forest.reduplication.rho | post-hoc, forest reduplication, SD 3.5 |
| -0.38 | exp03.posthoc.models.sd35.per_prompt_per_field.p6_forest.fragmentation.rho | post-hoc, forest fragmentation, SD 3.5 |
| 0.25 | fig4.data.fraction_nonzero.Qwen3-VL-32B.reduplication | judge screen: 32B non-zero share, reduplication and distortion |
| 0.11 | fig4.data.fraction_nonzero.Qwen3-VL-32B.condensation | judge screen: 32B non-zero share, condensation |
| 0.04 | fig4.data.fraction_nonzero.Qwen3-VL-32B.fragmentation | judge screen: 32B non-zero share, fragmentation |
| 418 | exp03.sdxl.report.notes.n_conditioned | SDXL conditioned images in the primary analysis |
| 417 | exp03.sd35.report.notes.n_conditioned | SD 3.5 conditioned images in the primary analysis |
| 0.710 | exp03.sdxl.report.reliability.per_field.reduplication.gwet_ac2 | SDXL AC2 range, low end (reduplication) |
| 0.924 | exp03.sdxl.report.reliability.per_field.fragmentation.gwet_ac2 | SDXL AC2 range, high end (fragmentation) |
| 0.803 | exp03.sd35.report.reliability.per_field.distortion.gwet_ac2 | SD 3.5 AC2 range, low end (distortion) |
| 0.953 | exp03.sd35.report.reliability.per_field.fragmentation.gwet_ac2 | SD 3.5 AC2 range, high end (fragmentation) |
| 0.376 | exp03.sdxl.report.reliability.per_field.distortion.percent_agreement | SDXL percent agreement, low end (distortion) |
| 0.724 | exp03.sdxl.report.reliability.per_field.fragmentation.percent_agreement | SDXL percent agreement, high end (fragmentation) |
| 0.377 | exp03.sd35.report.reliability.per_field.distortion.percent_agreement | SD 3.5 percent agreement, low end (distortion) |
| 0.824 | exp03.sd35.report.reliability.per_field.fragmentation.percent_agreement | SD 3.5 percent agreement, high end (fragmentation) |
| 90 | allow:analysis.md section 4, approximate count of truncated Llama replies | Llama truncated-JSON replies |
| 400 | allow:harness setting before the 2026-08-08 repair (max_tokens) | Llama decode cap that truncated replies |
| -0.158 | exp03.sensitivity2.S3_model_by_guidance_interaction.z_composite_prereg.interaction_sdxl_minus_sd35 | exploratory S3: model x guidance interaction, z-composite |
| -0.067 | exp03.sensitivity2.S3_model_by_guidance_interaction.z_composite_prereg.ci95[1] | S3 interaction CI upper, z-composite |
| -0.164 | exp03.sensitivity2.S3_model_by_guidance_interaction.raw_composite.interaction_sdxl_minus_sd35 | S3 interaction, raw judge-mean composite |
| -0.222 | exp03.sensitivity2.S3_model_by_guidance_interaction.raw_composite.ci95[0] | S3 interaction CI lower, raw composite |
| -0.105 | exp03.sensitivity2.S3_model_by_guidance_interaction.raw_composite.ci95[1] | S3 interaction CI upper, raw composite |
| 835 | exp03.sensitivity2.S3_model_by_guidance_interaction.z_composite_prereg.n | conditioned images pooled over both models |
| -0.337 | exp03.sensitivity2.models.sd35.S4_random_slopes_by_prompt.ci95[0] | S4 random slope by prompt, SD 3.5 CI lower |
| -0.027 | exp03.sensitivity2.models.sd35.S4_random_slopes_by_prompt.ci95[1] | S4 random slope by prompt, SD 3.5 CI upper |
| -0.448 | exp03.sensitivity2.models.sdxl.S4_random_slopes_by_prompt.ci95[0] | S4 random slope by prompt, SDXL CI lower |
| -0.231 | exp03.sensitivity2.models.sdxl.S4_random_slopes_by_prompt.ci95[1] | S4 random slope by prompt, SDXL CI upper |
| 0.112 | exp03.sensitivity2.models.sdxl.S4_random_slopes_by_prompt.random_slope_sd | S4 random-slope SD, SDXL |
| 0.177 | exp03.sensitivity2.models.sd35.S4_random_slopes_by_prompt.random_slope_sd | S4 random-slope SD, SD 3.5 |
| -0.322 | exp03.sensitivity2.models.sd35.S5_cluster_bootstrap.by_prompt.lmm_slope_ci95[0] | S5 prompt-cluster bootstrap slope CI lower, SD 3.5 |
| -0.050 | exp03.sensitivity2.models.sd35.S5_cluster_bootstrap.by_prompt.lmm_slope_ci95[1] | S5 prompt-cluster bootstrap slope CI upper, SD 3.5 |
| -0.407 | exp03.sensitivity2.models.sd35.S5_cluster_bootstrap.by_prompt.spearman_ci95[0] | S5 prompt-cluster Spearman CI lower, SD 3.5 |
| -0.001 | exp03.sensitivity2.models.sd35.S5_cluster_bootstrap.by_prompt.spearman_ci95[1] | S5 prompt-cluster Spearman CI upper, SD 3.5 |
| -0.238 | exp03.sensitivity2.models.sd35.S5_cluster_bootstrap.by_seed.lmm_slope_ci95[0] | S5 seed-cluster bootstrap slope CI lower, SD 3.5 |
| -0.125 | exp03.sensitivity2.models.sd35.S5_cluster_bootstrap.by_seed.lmm_slope_ci95[1] | S5 seed-cluster bootstrap slope CI upper, SD 3.5 |
| 0.271 | exp03.sensitivity2.S6_model_vs_model_on_human_subset.mean_per_field_kappa | S6 Claude vs Qwen3-VL-32B mean per-field kappa on the 28 |
| 0.056 | exp03.sensitivity2.S6_model_vs_model_on_human_subset.composite_bootstrap.ci95[0] | S6 model vs model CI lower |
| 0.448 | exp03.sensitivity2.S6_model_vs_model_on_human_subset.composite_bootstrap.ci95[1] | S6 model vs model CI upper |
| 0.069 | exp03.sensitivity2.S6_model_vs_model_on_human_subset.weighted_kappa.distortion | S6 judge vs judge distortion kappa on the 28 |
| 835 | derived:exp03.posthoc.models.sdxl.n_conditioned+exp03.posthoc.models.sd35.n_conditioned | conditioned images with complete replies |
| 20 | allow:design, 10 empty-prompt baselines per model x 2 | baselines analysed |
| 855 | derived:exp03.posthoc.models.sdxl.n_conditioned+exp03.posthoc.models.sd35.n_conditioned+20 | analysed images with complete replies |
| 0.740 | exp03.sensitivity2.models.sdxl.S7_trimmed_arm_balance.residual_Q_smd_high_minus_low | S7 residual quality SMD between trimmed arms, SDXL |
| 0.284 | exp03.sensitivity2.models.sd35.S7_trimmed_arm_balance.residual_Q_smd_high_minus_low | S7 residual quality SMD between trimmed arms, SD 3.5 |
| -0.570 | exp03.sensitivity2.models.sdxl.S8_partial_rho_prompt_bootstrap.ci95_prompt_cluster[0] | S8 prompt-cluster bootstrap of the partial rho, SDXL CI lower |
| -0.251 | exp03.sensitivity2.models.sdxl.S8_partial_rho_prompt_bootstrap.ci95_prompt_cluster[1] | S8 prompt-cluster bootstrap of the partial rho, SDXL CI upper |
| -0.410 | exp03.sensitivity2.models.sd35.S8_partial_rho_prompt_bootstrap.ci95_prompt_cluster[0] | S8 prompt-cluster bootstrap of the partial rho, SD 3.5 CI lower |
| 0.024 | exp03.sensitivity2.models.sd35.S8_partial_rho_prompt_bootstrap.ci95_prompt_cluster[1] | S8 prompt-cluster bootstrap of the partial rho, SD 3.5 CI upper |
| 2.172 | exp03.sd35.report.baseline_composite_uncond | Figure 2 as numbers |
| 1.094 | exp03.sd35.report.composite_mean_by_guidance.1.0 | Figure 2 as numbers |
| -0.165 | exp03.sd35.report.composite_mean_by_guidance.11.0 | Figure 2 as numbers |
| -0.082 | exp03.sd35.report.composite_mean_by_guidance.15.0 | Figure 2 as numbers |
| -0.151 | exp03.sd35.report.composite_mean_by_guidance.2.0 | Figure 2 as numbers |
| -0.166 | exp03.sd35.report.composite_mean_by_guidance.3.0 | Figure 2 as numbers |
| -0.260 | exp03.sd35.report.composite_mean_by_guidance.5.0 | Figure 2 as numbers |
| -0.280 | exp03.sd35.report.composite_mean_by_guidance.7.0 | Figure 2 as numbers |
| -0.591 | exp03.sd35.report.quality_mean_by_guidance.1.0 | Figure 2 as numbers |
| 0.364 | exp03.sd35.report.quality_mean_by_guidance.11.0 | Figure 2 as numbers |
| -0.043 | exp03.sd35.report.quality_mean_by_guidance.15.0 | Figure 2 as numbers |
| -0.153 | exp03.sd35.report.quality_mean_by_guidance.2.0 | Figure 2 as numbers |
| -0.008 | exp03.sd35.report.quality_mean_by_guidance.3.0 | Figure 2 as numbers |
| 0.118 | exp03.sd35.report.quality_mean_by_guidance.5.0 | Figure 2 as numbers |
| 0.312 | exp03.sd35.report.quality_mean_by_guidance.7.0 | Figure 2 as numbers |
| 1.510 | exp03.sdxl.report.baseline_composite_uncond | Figure 2 as numbers |
| 1.108 | exp03.sdxl.report.composite_mean_by_guidance.1.0 | Figure 2 as numbers |
| -0.340 | exp03.sdxl.report.composite_mean_by_guidance.11.0 | Figure 2 as numbers |
| -0.350 | exp03.sdxl.report.composite_mean_by_guidance.15.0 | Figure 2 as numbers |
| 0.224 | exp03.sdxl.report.composite_mean_by_guidance.2.0 | Figure 2 as numbers |
| -0.051 | exp03.sdxl.report.composite_mean_by_guidance.3.0 | Figure 2 as numbers |
| -0.242 | exp03.sdxl.report.composite_mean_by_guidance.5.0 | Figure 2 as numbers |
| -0.360 | exp03.sdxl.report.composite_mean_by_guidance.7.0 | Figure 2 as numbers |
| -0.691 | exp03.sdxl.report.quality_mean_by_guidance.1.0 | Figure 2 as numbers |
| 0.291 | exp03.sdxl.report.quality_mean_by_guidance.11.0 | Figure 2 as numbers |
| 0.438 | exp03.sdxl.report.quality_mean_by_guidance.15.0 | Figure 2 as numbers |
| -0.127 | exp03.sdxl.report.quality_mean_by_guidance.2.0 | Figure 2 as numbers |
| -0.112 | exp03.sdxl.report.quality_mean_by_guidance.3.0 | Figure 2 as numbers |
| 0.049 | exp03.sdxl.report.quality_mean_by_guidance.5.0 | Figure 2 as numbers |
| 0.156 | exp03.sdxl.report.quality_mean_by_guidance.7.0 | Figure 2 as numbers |
| 60 | allow:design, 6 prompts x 10 seeds per guidance value | images per cell |
| 5000 | allow:design, bootstrap and permutation resamples (numbers_allow.txt) | resample count |
| 0.394 | exp03.sd35.report.reliability_conditioned_only.composite_weighted_kappa | SD 3.5 kappa, conditioned images only |
| 0.542 | exp03.sdxl.report.reliability_conditioned_only.composite_weighted_kappa | SDXL kappa, conditioned images only |
| 0.463 | exp03b.axes.models.sdxl.P1_veridicality_lmm.slope_standardized | Exp 03b P1, SDXL |
| 0.382 | exp03b.axes.models.sdxl.P1_veridicality_lmm.ci95[0] | Exp 03b P1 CI lower, SDXL |
| 0.543 | exp03b.axes.models.sdxl.P1_veridicality_lmm.ci95[1] | Exp 03b P1 CI upper, SDXL |
| 0.417 | exp03b.axes.models.sdxl.P2_dissociation.partial_spearman_controlling_kluver | Exp 03b P2, SDXL |
| 0.322 | exp03b.axes.models.sdxl.P2_dissociation.ci95[0] | Exp 03b P2 CI lower, SDXL |
| 0.506 | exp03b.axes.models.sdxl.P2_dissociation.ci95[1] | Exp 03b P2 CI upper, SDXL |
| 0.529 | exp03b.axes.models.sdxl.P3_reliability.composite_weighted_kappa | Exp 03b P3, SDXL |
| -0.261 | exp03b.axes.models.sd35.P1_veridicality_lmm.slope_standardized | Exp 03b P1, SD 3.5 |
| -0.347 | exp03b.axes.models.sd35.P1_veridicality_lmm.ci95[0] | Exp 03b P1 CI lower, SD 3.5 |
| -0.174 | exp03.sd35.report.judge_only_lmm.claude.slope_standardized | judge-only Claude slope, SD 3.5 (also Exp 03b P1 CI upper) |
| -0.161 | exp03b.axes.models.sd35.P2_dissociation.partial_spearman_controlling_kluver | Exp 03b P2, SD 3.5 |
| -0.262 | exp03b.axes.models.sd35.P2_dissociation.ci95[0] | Exp 03b P2 CI lower, SD 3.5 |
| -0.053 | exp03b.axes.models.sd35.P2_dissociation.ci95[1] | Exp 03b P2 CI upper, SD 3.5 |
| 0.485 | exp03b.axes.models.sd35.P3_reliability.composite_weighted_kappa | Exp 03b P3, SD 3.5 |
| -0.321 | exp03.sdxl.report.judge_only_lmm.claude.slope_standardized | judge-only Claude slope, SDXL |
| -0.295 | exp03.sdxl.report.judge_only_lmm.qwen.slope_standardized | judge-only Qwen slope, SDXL |
| -0.141 | exp03.sd35.report.judge_only_lmm.qwen.slope_standardized | judge-only Qwen slope, SD 3.5 |
| -0.155 | exp03.qpaths.models.sdxl.quality_component_check.rho_clip_iqa_vs_aesthetic | CLIP-IQA vs aesthetic, SDXL |
| +0.170 | exp03.qpaths.models.sd35.quality_component_check.rho_clip_iqa_vs_aesthetic | CLIP-IQA vs aesthetic, SD 3.5 |
| 58 | exp03.sd35.report.n_by_guidance.2.0 | complete-case n at g = 2, SD 3.5 |
| 59 | exp03.sdxl.report.n_by_guidance.5.0 | complete-case n at g = 5, SDXL |
| 380 | exp03.sdxl.report3judge.reliability.n_images | three-judge complete cases, SDXL |
| 387 | exp03.sd35.report3judge.reliability.n_images | three-judge complete cases, SD 3.5 |
| 428 | exp03.sdxl.report.reliability.n_images | amended-panel complete cases including baselines, SDXL |
| 427 | exp03.sd35.report.reliability.n_images | amended-panel complete cases including baselines, SD 3.5 |
| 0.80 | allow:design, Exp 02 positive-control sensitivity threshold (exp02 preregistration, positive_control thresholds.validated, stored as text) | gate |
| 45% | derived:100*exp02.report.positive_control.photo_negatives.rate | abstract: Exp 02 ordinary-image flag rate |
| 20% | allow:design, Exp 02 registered ordinary-image ceiling 0.20 as a percent (exp02 preregistration thresholds, text) | abstract |
| 80% | allow:design, Exp 02 positive-control sensitivity threshold 0.80 as a percent (exp02 preregistration thresholds, text) | gate |
| 38% | derived:100*exp03.sdxl.report.reliability.per_field.distortion.percent_agreement | percent agreement low end (0.376 SDXL, 0.377 SD 3.5) |
| 72% | derived:100*exp03.sdxl.report.reliability.per_field.fragmentation.percent_agreement | SDXL percent agreement high end |
| 82% | derived:100*exp03.sd35.report.reliability.per_field.fragmentation.percent_agreement | SD 3.5 percent agreement high end |
| 40% | derived:100*exp03.posthoc.models.sdxl.tiling_rate.p6_forest.at_lowest_g | forest tiling rate at lowest g, SDXL |
| 18% | derived:100*fig4.data.fraction_nonzero.Llama-3.2-11B.reduplication | judge screen: Llama non-zero share |
| 36% | derived:100*fig4.data.fraction_nonzero.Gemma-3-27B.reduplication | judge screen: Gemma reduplication |
| 29% | derived:100*fig4.data.fraction_nonzero.Gemma-3-27B.distortion | judge screen: Gemma distortion |
| 25% | derived:100*fig4.data.fraction_nonzero.Qwen3-VL-32B.reduplication | judge screen: 32B reduplication and distortion |
| 11% | derived:100*fig4.data.fraction_nonzero.Qwen3-VL-32B.condensation | judge screen: 32B condensation |
| 4% | derived:100*fig4.data.fraction_nonzero.Qwen3-VL-32B.fragmentation | judge screen: 32B fragmentation, one image in 28 |
| 100% | derived:100*exp02.report.positive_control.positives.rate | Exp 02 control: positives rate |
| 0% | derived:100*exp02.report.positive_control.blank_negatives.rate | Exp 02 control: blanks rate |
| 9% | derived:100*exp02.report.positive_control.blank_negatives.ci95[1] | Exp 02 control: blanks CI upper |
| 31% | derived:100*exp02.report.positive_control.photo_negatives.ci95[0] | Exp 02 control: ordinary-image CI lower |
| -0.463 | exp03.sdxl.report.deconfound.raw_spearman_composite_vs_g | SDXL composite vs g before quality adjustment |
| -0.216 | exp03.sd35.report.deconfound.raw_spearman_composite_vs_g | SD 3.5 composite vs g before quality adjustment |
| 420 | allow:design, 6 prompts x 7 g x 10 seeds conditioned images per model (exp03.prereg) | design count |
| 0.567 | exp03.humanhuman.composite_weighted_kappa | author vs second rater, composite kappa (exploratory) |
| 0.330 | exp03.humanhuman.composite_bootstrap.ci95[0] | human-human CI lower |
| 0.721 | exp03.humanhuman.composite_bootstrap.ci95[1] | human-human CI upper |
| 0.426 | exp03.rater2judges.rater2.human_vs_claude.composite_weighted_kappa | second rater vs Claude |
| 0.208 | exp03.rater2judges.rater2.human_vs_claude.composite_bootstrap.ci95[0] | second rater vs Claude CI lower |
| 0.575 | exp03.rater2judges.rater2.human_vs_claude.composite_bootstrap.ci95[1] | second rater vs Claude CI upper |
| 0.169 | exp03.rater2judges.rater2.human_vs_qwen.composite_weighted_kappa | second rater vs Qwen |
| -0.029 | exp03.rater2judges.rater2.human_vs_qwen.composite_bootstrap.ci95[0] | second rater vs Qwen CI lower |
| 0.391 | exp03.rater2judges.rater2.human_vs_qwen.composite_bootstrap.ci95[1] | second rater vs Qwen CI upper |
