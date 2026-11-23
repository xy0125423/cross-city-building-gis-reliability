# Recorded reference results

Use `results/P5_02/tables/architecture_paired_route_metrics.csv` for 24 formal
architecture/seed/route results. IoU/Precision/Recall/F1 are fractions, not percentages.
Native per-scene and projected unique-support area summaries have different
denominators and must not be equated.

Use `results/P5_03/tables/all_288_spatial_summaries.csv`,
`all_288_main_scenario.csv`, `all_288_27_scenarios.csv`,
`origin_sensitivity_ranges.csv` and `zero_hidden_origin_stability.csv` for the
288-combination GIS experiment and threshold/origin diagnostics. Strata mean that
summary table row counts can exceed the number of experiment combinations.

The formal main figures are the five `fig1`–`fig5` originals in `figures/`; three
`figS` supplements are retained too. The PDFs/PNGs/SVGs are recorded research outputs,
not results rerun during release construction. Original source byte hashes for
selected published assets appear in `PROVENANCE.json`.

The reproducibility gate is currently a synthetic functional test, not statistical
agreement with the formal tables. No numerical tolerance for a full rerun is
claimed as passed before observing that rerun. In particular, do not infer that
greater IoU universally reduces FP/FN spatial error or that all cross-city models
underestimate building area.
