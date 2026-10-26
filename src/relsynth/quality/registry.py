"""Definitions of every metric in the 2.0 quality report.

docs/metrics.md is generated from this table (docs/make_metrics_reference.py),
and every check in a report names its metric by these ids.  Fields:

area        fidelity, utility or integrity (privacy metrics belong to T06 and
            are passed through from the 1.0 scorecard without an area)
scope       column, table, relationship, path or database
direction   "lower" (0 or `center` is ideal) or "higher"
center      value of the metric when real and synthetic laws are equal
tolerance   practical floor of the limit: a difference below it is accepted
            however large the data (a guess stated as such; set your own with
            the quality config `tolerances`)
null        how the limit is calibrated
"""
from __future__ import annotations

SPLIT = "real-vs-real splits (Prop. 106), rescaled to the compared sizes"

METRICS = {
    # ------------------------------------------------------------ single column
    "w1_raw_sd": dict(
        area="fidelity", scope="column", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="Wasserstein-1 distance, raw scale",
        definition="W1(P_real, P_syn) of the non-null values of a numeric, date or datetime column "
                   "(dates in days), divided by the standard deviation of the real values.",
        range="[0, inf); 0 = equal laws", reference="Villani 2009; Panaretos & Zemel 2019"),
    "w1_rank_normal": dict(
        area="fidelity", scope="column", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="Wasserstein-1 distance, rank-normal scale",
        definition="W1 between the normal scores z = Phi^-1(F_real(x)) of the real and synthetic "
                   "non-null values, with F_real the real mid-ECDF on n+1 points. Invariant to "
                   "monotone transformations; a synthetic value beyond the real range counts as the "
                   "most extreme real score (the raw-scale W1 sees the excess).",
        range="[0, inf); 0 = equal laws", reference="Bickel & Freedman 1981; docs/proofs Section v1.0 (normal scores)"),
    "js_distance": dict(
        area="fidelity", scope="column", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="Jensen-Shannon distance",
        definition="Square root of the Jensen-Shannon divergence (base 2) between the real and synthetic "
                   "distributions of a column's levels (null is a level); numeric columns on 20 real "
                   "quantile bins.",
        range="[0, 1]; 0 = equal laws", reference="Lin 1991; Endres & Schindelin 2003"),
    # ------------------------------------------------------------ single table
    "kway2_tvd_mean": dict(
        area="fidelity", scope="table", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="Mean 2-way marginal TVD",
        definition="Mean over all pairs of attribute columns of the total variation distance between the "
                   "real and synthetic joint distributions of the discretised pair (categoricals: 20 most "
                   "frequent real levels, the rest pooled; numerics: real quintiles; null a level).",
        range="[0, 1]", reference="NIST CRC SDNist k-marginal score (Task et al. 2023)"),
    "kway2_tvd_max": dict(
        area="fidelity", scope="table", direction="lower", center=0.0, tolerance=0.15, null=SPLIT,
        name="Largest 2-way marginal TVD",
        definition="Largest of the 2-way marginal TVDs above (the worst pair is named in the report).",
        range="[0, 1]", reference="NIST CRC SDNist"),
    "kway3_tvd_mean": dict(
        area="fidelity", scope="table", direction="lower", center=0.0, tolerance=0.08, null=SPLIT,
        name="Mean 3-way marginal TVD",
        definition="Mean TVD over all 3-way marginals of the discretised columns, or over a seeded "
                   "random set of 50 triples when there are more.",
        range="[0, 1]", reference="NIST CRC SDNist"),
    "alpha_precision_gap": dict(
        area="fidelity", scope="table", direction="lower", center=0.0, tolerance=0.10, null=SPLIT,
        name="alpha-precision gap",
        definition="1 - I_P with I_P = 1 - 2 mean_a |P(a) - a| over a = 0.05..1, where P(a) is the share "
                   "of synthetic rows inside the real a-support (the ball around the real mean, in "
                   "rank-normal feature space, holding a share a of the real rows).",
        range="[0, 1]; 0 = ideal", reference="Alaa et al., ICML 2022; Qian et al. Synthcity, NeurIPS 2023"),
    "beta_recall_gap": dict(
        area="fidelity", scope="table", direction="lower", center=0.0, tolerance=0.10, null=SPLIT,
        name="beta-recall gap",
        definition="1 - I_R with I_R = 1 - 2 mean_b |R(b) - b|, where R(b) is the share of real rows whose "
                   "nearest synthetic row lies within the real row's 5-NN radius among real rows and "
                   "inside the synthetic b-support.",
        range="[0, 1]; 0 = ideal", reference="Alaa et al., ICML 2022"),
    # ------------------------------------------------------------ utility
    "tstr_gap": dict(
        area="utility", scope="table", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="TSTR gap",
        definition="TRTR - TSTR: score of a model trained on real rows minus that of the same learner "
                   "trained on synthetic rows, both tested on held-out real rows. Learners: L2 logistic "
                   "(weighted one-vs-rest AUC) for a categorical target, ridge on the rank-normal target "
                   "(R^2) for a numeric one. One-sided: a synthetic set that trains a better model passes.",
        range="[-1, 1]", reference="Esteban et al. 2017; Jordon et al. 2022"),
    "trts_gap": dict(
        area="utility", scope="table", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="TRTS gap",
        definition="|TRTR - TRTS|: a model trained on real rows scores the same on synthetic rows as on "
                   "held-out real rows. Two-sided: synthetic rows that are easier (too regular) also fail.",
        range="[0, 1]", reference="Esteban et al. 2017"),
    "mt_tstr_gap": dict(
        area="utility", scope="table", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="Multi-table TSTR gap",
        definition="TSTR gap of a parent-table target predicted from the parent's other columns and the "
                   "aggregates of its child tables (count, numeric means, distinct counts).",
        range="[-1, 1]", reference="Hudovernik et al. TMLR 2026 (utility on aggregated tables)"),
    "mt_trts_gap": dict(
        area="utility", scope="table", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="Multi-table TRTS gap",
        definition="TRTS gap of the multi-table task above.", range="[0, 1]", reference="Esteban et al. 2017"),
    # ------------------------------------------------------------ multi-table
    "fanout_w1_rel": dict(
        area="fidelity", scope="relationship", direction="lower", center=0.0, tolerance=0.10, null=SPLIT,
        name="Fan-out law W1",
        definition="W1 between the real and synthetic laws of the number of child rows per parent row, "
                   "divided by the real mean count.",
        range="[0, inf)", reference="Hudovernik et al. TMLR 2026 (cardinality); Villani 2009"),
    "fanout_zero_share_diff": dict(
        area="fidelity", scope="relationship", direction="lower", center=0.0, tolerance=0.03, null=SPLIT,
        name="Childless-parent share difference",
        definition="|share of parents without children, real - synthetic|.", range="[0, 1]", reference="-"),
    "fanout_dispersion_log_ratio": dict(
        area="fidelity", scope="relationship", direction="lower", center=0.0, tolerance=0.25, null=SPLIT,
        name="Fan-out dispersion log ratio",
        definition="|log(D_syn / D_real)| with D = variance / mean of the per-parent child counts "
                   "(1 for Poisson counts; over-dispersion is what block models must reproduce).",
        range="[0, inf)", reference="Cameron & Trivedi 2013"),
    "fanout2_w1_rel": dict(
        area="fidelity", scope="path", direction="lower", center=0.0, tolerance=0.15, null=SPLIT,
        name="Two-hop fan-out law W1",
        definition="Fan-out law W1 of grandchild rows per grandparent row along a two-hop path.",
        range="[0, inf)", reference="-"),
    "hop_trend_gap": dict(
        area="fidelity", scope="path", direction="lower", center=0.0, tolerance=0.05, null=SPLIT,
        name="k-hop trend gap",
        definition="1 - mean over column pairs (one column of the bottom table, one of the top table of a "
                   "1- or 2-hop foreign-key path, on the denormalised rows) of the SDMetrics column-pair "
                   "trend score: 1 - |r_real - r_syn|/2 (Pearson) for two numeric columns, 1 - TVD of the "
                   "joint table (numerics in 10 equal-width bins) otherwise.",
        range="[0, 1]", reference="Pang et al. ClavaDDPM, NeurIPS 2024; Hudovernik et al. TMLR 2026"),
    "hop_trend_worst_gap": dict(
        area="fidelity", scope="path", direction="lower", center=0.0, tolerance=0.15, null=SPLIT,
        name="Worst k-hop pair gap",
        definition="1 - the smallest column-pair trend score of the path (the pair is named).",
        range="[0, 1]", reference="as hop_trend_gap"),
    "dda_accuracy": dict(
        area="fidelity", scope="table", direction="lower", center=0.5, tolerance=0.60, null=SPLIT,
        name="Aggregation detection accuracy (C2ST-Agg)",
        definition="5-fold stratified cross-validated accuracy of a classifier separating balanced samples "
                   "(at most 5,000 each) of real and synthetic parent rows, each augmented per child "
                   "relationship with the child count, the means of numeric child columns and the distinct "
                   "counts of categorical child columns. Default classifier: L2 logistic regression on "
                   "rank-normal scores, squares, null indicators and one-hot levels; 'xgboost' as SyntheRela.",
        range="[0, 1]; 0.5 = indistinguishable", reference="Hudovernik et al. TMLR 2026; Lopez-Paz & Oquab 2017"),
}

# 1.0 scorecard items by area (everything else is fidelity)
SCORECARD_AREAS = {
    "fk_integrity": "integrity", "rule_violation_rate": "integrity", "unique_duplicate_rate": "integrity",
    "tree_cycles": "integrity", "rows_removed_share": "integrity",
    "text_tstr_ratio": "utility",
    "dcr_ratio": "privacy", "exact_match_rate": "privacy", "text_rare_token_leaks": "privacy",
    "text_rare_ngram_copies": "privacy", "text_rare_exact_copies": "privacy",
}

AREAS = ("fidelity", "utility", "integrity")
