"""R1: per-channel split conformal sets, followed by hard review gates.

    Marginal set coverage is NOT singleton accuracy. Exchangeability within each
    channel is an external assumption. Time checks do not establish that assumption.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import math

from .common import digest, fields, integer, number, require, token


def probabilities(values, labels):
    require(isinstance(values, dict) and set(values) == set(labels), "PROBABILITY_LABELS")
    for p in values.values():
        number(p, "PROBABILITY_VALUE", 0, 1)
    require(math.isclose(sum(values.values()), 1, abs_tol=1e-9, rel_tol=0), "PROBABILITY_SUM")


def quantile(scores, alpha):
    require(bool(scores), "EMPTY_CALIBRATION")
    number(alpha, "ALPHA", 0, 1)
    require(0 < alpha < 1, "ALPHA")
    for value in scores:
        number(value, "SCORE", 0, 1)
    # Decimal avoids ceil(3.0000000000000004) off-by-one at decimal boundaries.
    rank = int(((len(scores) + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return (sorted(scores)[rank - 1] if rank <= len(scores) else 1.0), rank


@dataclass(frozen=True)
class Calibration:
    labels: tuple[str, ...]
    model_version: str
    taxonomy_version: str
    cutoff: int
    alpha: float
    min_group: int
    groups: tuple[tuple[str, int, float, int], ...]
    calibration_ids: tuple[str, ...]
    training_ids: tuple[str, ...]
    data_digest: str

    @property
    def version(self):
        return digest(asdict(self))


def fit(rows, *, labels, model_version, taxonomy_version, cutoff, training_ids=(), alpha=.1, min_group=10):
    labels = tuple(labels)
    require(len(labels) >= 2 and len(set(labels)) == len(labels), "LABELS")
    for label in labels:
        token(label)
    token(model_version)
    token(taxonomy_version)
    integer(cutoff, "CUTOFF")
    integer(min_group, "MIN_GROUP", 1)
    require(bool(rows), "EMPTY_CALIBRATION")
    train = set(training_ids)
    for item in train:
        token(item)
    seen, scores = set(), defaultdict(list)
    for row in rows:
        fields(row, ("id", "channel", "model_version", "taxonomy_version", "observed_at", "truth", "probabilities"), "CALIBRATION_FIELDS")
        token(row["id"])
        token(row["channel"])
        require(row["id"] not in seen | train, "CALIBRATION_LEAKAGE_OR_DUPLICATE")
        seen.add(row["id"])
        require(row["model_version"] == model_version and row["taxonomy_version"] == taxonomy_version,
                "CALIBRATION_VERSION")
        integer(row["observed_at"], "OBSERVED_AT")
        require(row["observed_at"] <= cutoff, "FUTURE_CALIBRATION")
        require(row["truth"] in labels, "TRUTH_LABEL")
        probabilities(row["probabilities"], labels)
        scores[row["channel"]].append(1 - row["probabilities"][row["truth"]])
    groups = tuple((channel, len(values), *quantile(values, alpha)) for channel, values in sorted(scores.items()))
    return Calibration(labels, model_version, taxonomy_version, cutoff, alpha, min_group, groups,
                       tuple(sorted(seen)), tuple(sorted(train)), digest(sorted(rows, key=lambda r: r["id"])))


def route(calibration, row, *, manual_only=False):
    """Input contains no truth label. Output is a draft classification or review."""
    fields(row, ("id", "channel", "observed_at", "eligible", "risk_flag", "drift_flag", "probabilities", "model_version", "taxonomy_version"), "PREDICTION_FIELDS")
    token(row["id"])
    token(row["channel"])
    integer(row["observed_at"], "OBSERVED_AT")
    require(row["id"] not in calibration.calibration_ids + calibration.training_ids, "INFERENCE_LEAKAGE")
    require(row["observed_at"] > calibration.cutoff, "INFERENCE_TIME")
    for flag in ("eligible", "risk_flag", "drift_flag"):
        require(type(row[flag]) is bool, "POLICY_FLAG")
    require(type(manual_only) is bool, "MANUAL_FLAG")
    p = row["probabilities"]
    if p is not None:
        probabilities(p, calibration.labels)
    groups = {c: (n, q, rank) for c, n, q, rank in calibration.groups}
    reasons, prediction_set = [], None
    if manual_only:
        reasons.append("MANUAL_ONLY")
    if not row["eligible"]:
        reasons.append("INELIGIBLE_FOR_AUTO")
    if row["risk_flag"]:
        reasons.append("RISK_REVIEW")
    if row["drift_flag"]:
        reasons.append("DECLARED_DRIFT")
    versions_match = (row["model_version"] == calibration.model_version and
                      row["taxonomy_version"] == calibration.taxonomy_version)
    if not versions_match:
        reasons.append("VERSION_MISMATCH")
    if p is None:
        reasons.append("NO_SCORES")
    if row["channel"] not in groups or groups[row["channel"]][0] < calibration.min_group:
        reasons.append("INSUFFICIENT_CHANNEL_CALIBRATION")
    elif versions_match and p is not None:
        q = groups[row["channel"]][1]
        prediction_set = [label for label in calibration.labels if 1 - p[label] <= q + 1e-12]
        if len(prediction_set) != 1:
            reasons.append("EMPTY_SET" if not prediction_set else "MULTIPLE_LABELS")
    return {"id": row["id"], "channel": row["channel"], "prediction_set": prediction_set,
            "decision": "REVIEW" if reasons else "AUTO_DRAFT", "reasons": reasons,
            "risk_priority": row["risk_flag"], "calibration_version": calibration.version,
            "executed_customer_action": False}


def run(calibration, rows, truths, *, review_capacity=3, manual_only=False):
    integer(review_capacity, "REVIEW_CAPACITY")
    for row in rows:
        fields(row, ("id",), "PREDICTION_FIELDS")
    ids = [r["id"] for r in rows]
    require(len(ids) == len(set(ids)) and bool(rows), "BATCH_IDS")
    require(set(truths) == set(ids) and all(y in calibration.labels for y in truths.values()), "EVALUATION_TRUTH")
    decisions = [route(calibration, row, manual_only=manual_only) for row in rows]
    reviews = sorted((d for d in decisions if d["decision"] == "REVIEW"),
                     key=lambda d: (not d["risk_priority"], d["id"]))
    auto = [d for d in decisions if d["decision"] == "AUTO_DRAFT"]
    evaluated = [d for d in decisions if d["prediction_set"] is not None]
    covered = sum(truths[d["id"]] in d["prediction_set"] for d in evaluated)
    wrong = sum(truths[d["id"]] not in d["prediction_set"] for d in auto)
    return {"project": "R1", "policy": "conformal-review-v1", "calibration": asdict(calibration),
            "calibration_version": calibration.version, "input_digest": digest(rows),
            "truth_digest": digest(truths), "manual_only": manual_only, "review_capacity": review_capacity,
            "decisions": decisions, "review_now": [d["id"] for d in reviews[:review_capacity]],
            "review_backlog": [d["id"] for d in reviews[review_capacity:]],
            "metrics": {"total": len(rows), "set_coverage_n": len(evaluated), "set_covered": covered,
                        "set_coverage": covered / len(evaluated) if evaluated else None,
                        "auto_n": len(auto), "auto_error_n": wrong,
                        "auto_error_rate": wrong / len(auto) if auto else None,
                        "review_n": len(reviews), "backlog_n": max(0, len(reviews) - review_capacity)},
            "scope": "synthetic score fixtures; marginal coverage does not guarantee selected accuracy",
            "trace": ["validate_split", "channel_quantile", "hard_review_gates", "capacity_plan", "offline_evaluation"]}
