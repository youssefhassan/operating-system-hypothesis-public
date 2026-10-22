# Author responses to the two xsci reviews of Preprint I v1 (2026-10-22)

Two AI referee reports were posted on xsci, each with its own author-response box.
Numbers come from `experiments/03_l23_hardening/posthoc_review_xsci.py` (pre-specified in
`posthoc_review_xsci_prespec.md` before running, output `posthoc_review_xsci.json`).
Exploratory; no registered gate changes.

---

# Response A: reviewer3.com report (XSci 499431ce, seven comments)

All seven comments are correct on the facts. I reran the analyses they ask for on 2026-10-22, with each check written down before running; the script and its output are in the public repository. None of them changes a verdict.

**1. Composite averages judges before z-scoring.** Correct. Recomputed with each judge's scores z-scored within judge and field before averaging. The SDXL slope moves from -0.340 [-0.400, -0.279] to -0.308 [-0.361, -0.254]; SD 3.5 from -0.182 [-0.248, -0.115] to -0.158 [-0.213, -0.102]. Both sit between the two single-judge slopes already reported (Claude alone -0.321 and -0.174, Qwen alone -0.295 and -0.141). The registered composite does lean toward Claude, by about 0.03 of slope. The verdicts are unchanged. The next version will state the weighting in the text and carry the within-judge composite in the sensitivity table.

**2. Interaction model random effects.** The published fit used (1|prompt) + (1|seed) shared across checkpoints, which the paper should have stated. Refitted with (1|prompt:model) + (1|seed:model) added: the SDXL minus SD 3.5 interaction is -0.158 [-0.248, -0.067] in the published specification and -0.158 [-0.248, -0.068] with checkpoint-specific intercepts. The formula will be given in full.

**3. Quality proxies controlled separately and together.** Partial Spearman of composite against guidance, controlling CLIP-IQA and the aesthetic score as two separate covariates: SDXL -0.501 [-0.569, -0.425] (image bootstrap; prompt-cluster bootstrap [-0.592, -0.364]), SD 3.5 -0.234 [-0.329, -0.132] (prompt-cluster [-0.429, +0.054]). Controlling each alone: CLIP-IQA -0.509 and -0.242, aesthetic -0.465 and -0.213; the registered averaged Q gives -0.433 and -0.246. The averaged proxy is the most conservative of the four on SDXL. These will go in the table.

**4. Forest control inside the 5-of-6 gate.** The gate as pre-registered counts six prompts, and the same registration marks the forest prompt as the control, so the tally in the paper is the registered one. The reviewer's point stands: restricted to the five test prompts, SDXL is 5/5 and SD 3.5 is 4/5, and on SD 3.5 the sixth negative sign is the control at -0.02, which is flat, not a sixth confirmation. Both tallies will be reported.

**5. Empty-prompt rubric.** The judges received a distinct variant for the baselines, which the paper omitted. The intended-scene sentence is replaced by: "This image was generated from an EMPTY prompt (the model's own prior, with no intended scene). There is no 'correct' content. Score only the phenomena listed below on whatever is visibly present." It is in `rubric.py` in the repository and will be added to the verbatim-prompt appendix. The reliability consequence is already in the paper: SD 3.5's kappa is 0.440 with the ten baselines and 0.394 on conditioned images alone, and I report both because the registered set includes the baselines and the conditioned-only number is the stricter reading.

**6. Quality adjustment inside the mixed model.** Guidance fixed effect with Q as a covariate, same random intercepts: SDXL -0.270 [-0.339, -0.201], SD 3.5 -0.157 [-0.225, -0.090]. With the two proxies as separate covariates: SDXL -0.237 [-0.313, -0.161], SD 3.5 -0.164 [-0.233, -0.094]. On SDXL the within-prompt adjustment removes about a third of the slope, more than the pooled partial correlation suggested, and what remains clears the registered -0.2 bar with the interval away from zero. That is the honest number for the quality question and it will replace the pooled partial as the headline adjustment, with the pooled value kept because it is what was registered.

**7. Screening images inside the corpus.** The 28 screening images are the human-rated subset (13 SDXL, 15 SD 3.5 among complete cases). Dropping them: SDXL slope -0.349 [-0.411, -0.287], partial rho -0.441, composite kappa 0.569 [0.521, 0.610] (conditioned only 0.549), tally 6/6 and 5/5; SD 3.5 slope -0.180 [-0.247, -0.113], partial rho -0.247, kappa 0.440 [0.376, 0.497] (conditioned only 0.392), tally 5/6 and 4/5. No change to the second decimal except the SDXL kappa, which rises.

Youssef Hassan

---

# Response B: xPeerd DAReview report (XSci 363923e4)

Thank you for a careful read. I accept the report's central recommendation: the paper should be read, and will be written, as an assay-development study with a substantive signal under an amended instrument. Point by point:

**Amended panel and confirmatory status (1, 7).** I agree with the recommendation and will adopt the wording: the SDXL result is confirmatory under an amended instrument, and the paper is an assay-development study with a substantive signal, not a settled validation of a construct. The abstract and conclusion will say so directly rather than disclosing the amendment and letting the reader weigh it.

**Human validity and construct language (2, 6).** Agreed. The construct the data support is judge-scored rubric movement; "object-level breakdown" will be confined to the discussion of what a larger human panel would test. The five phrasings in the introduction will be collapsed to one. A larger blinded human panel on the exact rubric is the first item in the next study and is not something I can add to this version.

**Competing explanations (3).** Prompt adherence was not measured and the report is right that this belongs in the central limitations, not in future work. The quality-adjusted mixed model above is the strongest adjustment the current data allow; it leaves the SDXL slope past the bar and leaves prompt adherence open. Image-text similarity will be scored on the existing images for v2, since the images and prompts are public and no new generation is needed.

**Linear endpoint versus a g = 1 step (4, recommendation 7).** Agreed. The log2 fit and the segmented reading are in the paper as post-hoc; v2 will move them into the main results with the exploratory label kept, and the SD 3.5 text will lead with the concentration at g = 1 rather than the slope.

**Architecture wording (5).** Agreed; "checkpoint" throughout, with the architecture confound stated where the two are compared.

**Forest control (6).** Agreed. With the control removed from the prompt-generality tally, SDXL is 5/5 and SD 3.5 is 4/5, and both tallies will be reported. The failed control will appear in the abstract alongside the gates that passed.

The minor style suggestions I will take selectively: the informal phrasings are deliberate, but the terminology standardisation and the typographic separation of registered from exploratory analyses are improvements and will be made.

Youssef Hassan

---

# Response C: XSci report (XSci 9bcf95ad)

Thank you. I agree with the positioning: the durable contribution is the instrument (the judge audit, the gradedness screen and the public corpus), and the next version will lead with it rather than with the dose-response. I ran the checks this report asks for on the public images on 2026-10-22, pre-specified before running; the script and output are in the public repository. The first changes how the SDXL result should be read.

Major 1, prompt adherence. Correct, and the most consequential point in any of the three reports. I computed CLIP ViT-L/14 image-text similarity against each registered prompt. It rises with guidance (Spearman +0.34 SDXL, +0.31 SD 3.5). The pooled partial correlation barely moves (-0.556 and -0.240 controlling similarity; -0.529 and -0.238 with both quality proxies added). Inside the mixed model, where the estimate is within prompt, it does move: the guidance slope is -0.248 [-0.312, -0.183] on SDXL and -0.082 [-0.148, -0.017] on SD 3.5 with similarity as a covariate, and -0.201 [-0.276, -0.127] and -0.082 [-0.150, -0.014] with similarity, CLIP-IQA and the aesthetic score together. So a within-prompt association survives on both checkpoints with intervals away from zero, but on SDXL it sits on the registered -0.2 bar rather than well past it, and on SD 3.5 more than half of it is shared with prompt adherence. One caveat: breakdown itself lowers image-text similarity, so this adjustment partly removes the outcome and is a lower bound, not a clean separation. The next version will report this in the main results and say that the composite is only partly separable from adherence.

Major 2, judge agreement. The suspicion is confirmed. Composite kappa on conditioned images within guidance bins: low (g 1-2) 0.550 SDXL and 0.419 SD 3.5; mid (g 3-5) 0.396 and 0.172; high (g 7-15) 0.220 and 0.190. The judges agree at the gate level where breakdown is common and weakly where it is rare, so the pooled 0.542 and 0.394 are carried by the low-guidance images. Together with Qwen3-VL-32B's unestablished human validity, this means the instrument gate licenses the low-guidance contrast more than a graded dose-response. The binned kappas will go next to the pooled ones, and a larger human panel stays the first item of the next study.

Major 3, literature. Agreed. MagicMirror (arXiv:2509.10260) and Kamali et al. (CHI 2025) will be engaged in Section 1.1, and the novelty claim restated against them: their taxonomies are larger and human-annotated; what this paper adds is a doseable perturbation, pre-registration and a judge-validity audit.

Minor comments. (3) and (5) are answered in my reply to the Reviewer3 report: on the five test prompts the tally is 5/5 and 4/5, and dropping the 28 screening images changes no slope or kappa beyond the second decimal; the reuse will be stated. (7) Dropping the seed random effect leaves both slopes unchanged (SDXL -0.340 [-0.402, -0.278], SD 3.5 -0.182 [-0.248, -0.116]). (4) The per-component adjustments will go into Table 4. (1) Agreed that the 20% ceiling was set without regard to prompts containing literal lattices; the Exp 01 reclassification follows the registered rule and stays, with that context stated. (2) and (11) Agreed; the rubric scores under-conditioning only, and without another degradation scored on it, "assay" is claimed for guidance and not for specificity. (6) Agreed; a non-inferiority formulation is what I would register now. (8), (9), (10) and (12) will be fixed: the clean-checkout regeneration claim reconciled, one list of Exp 02 gates, the Sonnet 4.6 and Sonnet 5 scales marked as not comparable, and compute and API cost reported.

Youssef Hassan
