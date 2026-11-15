"""R2: exact descriptive decomposition and reference-population standardization."""
from __future__ import annotations

from .common import digest, fields, integer, number, require, token


def validate(cells):
    require(isinstance(cells, dict) and bool(cells), "EMPTY_CELLS")
    result = {}
    for group, cell in cells.items():
        token(group)
        fields(cell, ("population", "invited", "promoter", "passive", "detractor"), "CELL_FIELDS")
        for key in ("population", "invited", "promoter", "passive", "detractor"):
            integer(cell[key], "CELL_COUNT")
        n = cell["promoter"] + cell["passive"] + cell["detractor"]
        require(n <= cell["invited"] <= cell["population"], "CELL_TOTAL")
        result[group] = {**cell, "responses": n,
                         "nps": 100 * (cell["promoter"] - cell["detractor"]) / n if n else None,
                         "response_rate": n / cell["invited"] if cell["invited"] else None}
    return result


def period(cells, weights, min_responses, min_response_rate):
    cells = validate(cells)
    n = sum(c["responses"] for c in cells.values())
    raw = 100 * sum(c["promoter"] - c["detractor"] for c in cells.values()) / n if n else None
    warnings = []
    for group, cell in cells.items():
        if cell["responses"] < min_responses:
            warnings.append(group + ":SMALL_CELL")
        if cell["response_rate"] is None or cell["response_rate"] < min_response_rate:
            warnings.append(group + ":LOW_RESPONSE")
    missing = [g for g, weight in weights.items() if weight > 0 and (g not in cells or not cells[g]["responses"])]
    standardized = None if missing else sum(weights[g] * cells[g]["nps"] for g in weights if weights[g] > 0)
    # Sum of per-respondent normalized weights is 1; Kish weight-only ESS is 1 / sum(w_i^2).
    # This is not a variance estimator, bias correction guarantee, or clustered survey ESS.
    weight_square_sum = None if missing else sum(weights[g] ** 2 / cells[g]["responses"]
                                               for g in weights if weights[g] > 0)
    ess = 1 / weight_square_sum if weight_square_sum else None
    return {"cells": cells, "responses": n, "raw_nps": raw, "standardized_nps": standardized,
            "weight_only_ess": ess, "missing_target_strata": missing, "warnings": warnings,
            "response_mix": {g: c["responses"] / n if n else 0 for g, c in cells.items()}}


def compare(before, after, target_counts, *, reference_version="population-v1", min_responses=10, min_response_rate=.2):
    token(reference_version)
    integer(min_responses, "MIN_RESPONSES", 1)
    number(min_response_rate, "MIN_RESPONSE_RATE", 0, 1)
    require(bool(target_counts), "EMPTY_REFERENCE")
    for group, count in target_counts.items():
        token(group)
        integer(count, "TARGET_COUNT")
    total = sum(target_counts.values())
    require(total > 0, "EMPTY_REFERENCE")
    require(set(before) <= set(target_counts) and set(after) <= set(target_counts), "UNKNOWN_STRATUM")
    weights = {g: c / total for g, c in sorted(target_counts.items())}
    first, second = (period(c, weights, min_responses, min_response_rate) for c in (before, after))
    common = set(before) == set(after) and all(first["cells"][g]["responses"] and second["cells"][g]["responses"] for g in before)
    decomposition = None
    if common:
        parts = []
        for g in sorted(before):
            w0, w1 = first["response_mix"][g], second["response_mix"][g]
            y0, y1 = first["cells"][g]["nps"], second["cells"][g]["nps"]
            parts.append({"stratum": g, "within": (w0 + w1) / 2 * (y1 - y0),
                          "mix": (w1 - w0) * (y0 + y1) / 2})
        within, mix = sum(p["within"] for p in parts), sum(p["mix"] for p in parts)
        delta = second["raw_nps"] - first["raw_nps"]
        require(abs(delta - within - mix) < 1e-9, "DECOMPOSITION_IDENTITY")
        decomposition = {"raw_delta": delta, "within": within, "mix": mix, "parts": parts,
                         "identity_residual": delta - within - mix}
    standard_delta = (second["standardized_nps"] - first["standardized_nps"]
                      if first["standardized_nps"] is not None and second["standardized_nps"] is not None else None)
    status = "DESCRIPTIVE_ONLY" if standard_delta is not None and decomposition is not None else "HOLD"
    return {"project": "R2", "policy": "nps-composition-v1", "status": status,
            "reference_version": reference_version, "reference_digest": digest(target_counts), "target_weights": weights,
            "input_digest": digest({"before": before, "after": after}),
            "min_responses": min_responses, "min_response_rate": min_response_rate,
            "before": first, "after": second, "decomposition": decomposition, "standardized_delta": standard_delta,
            "assumption": "within each target stratum respondents represent nonrespondents; not verified by weighting",
            "causal_claim_allowed": False,
            "trace": ["validate_aggregate_grain", "freeze_target_weights", "positivity_check", "decompose_or_hold"]}
