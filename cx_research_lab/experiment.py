"""R3: frozen experiment specification, assignment integrity, SRM and CUPED.

Fixed-horizon, two-arm Bernoulli 50/50 lab. CI uses a large-sample normal
approximation with a coefficient frozen on historical data. No sequential test,
missing-outcome imputation, cluster inference, or automatic rollout is provided.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from statistics import fmean, variance

from .common import digest, fields, integer, number, require, token


@dataclass(frozen=True)
class Cuped:
    theta: float
    x_mean: float
    fitted_before: int
    historical_n: int
    historical_digest: str
    metric: str
    feature_version: str

    @property
    def version(self):
        return digest(asdict(self))


def fit_cuped(history, *, fitted_before, metric="resolution_minutes", feature_version="prior_resolution_minutes-v1"):
    integer(fitted_before, "FIT_TIME")
    require(metric in ("resolution_minutes", "nps_points"), "METRIC")
    token(feature_version)
    require(len(history) >= 2, "HISTORY_TOO_SMALL")
    ids = set()
    for row in history:
        fields(row, ("id", "x", "y", "x_at", "y_at"), "HISTORY_FIELDS")
        token(row["id"])
        require(row["id"] not in ids, "HISTORY_DUPLICATE")
        ids.add(row["id"])
        number(row["x"], "HISTORICAL_VALUE", -1000000, 1000000)
        number(row["y"], "HISTORICAL_VALUE", -1000000, 1000000)
        require(row["y"] in (-100, 0, 100) if metric == "nps_points" else row["y"] >= 0, "HISTORICAL_METRIC")
        integer(row["x_at"], "HISTORICAL_TIME")
        integer(row["y_at"], "HISTORICAL_TIME")
        require(row["x_at"] < row["y_at"] < fitted_before, "HISTORICAL_LEAKAGE")
    xs, ys = [r["x"] for r in history], [r["y"] for r in history]
    mx, my = fmean(xs), fmean(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    theta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator if denominator else 0.0
    return Cuped(theta, mx, fitted_before, len(history), digest(sorted(history, key=lambda r: r["id"])), metric, feature_version)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    seed: str
    cohort: tuple[str, ...]
    registered_at: int
    start: int
    end: int
    cuped_version: str
    metric: str = "resolution_minutes"
    min_per_arm: int = 50
    srm_alpha: float = .001
    max_error_rate: float = .05

    def __post_init__(self):
        token(self.experiment_id)
        token(self.seed)
        require(type(self.cohort) is tuple and 1 <= len(self.cohort) <= 10000, "COHORT_SIZE")
        require(len(set(self.cohort)) == len(self.cohort), "COHORT_DUPLICATE")
        for item in self.cohort:
            token(item)
        for value in (self.registered_at, self.start, self.end):
            integer(value, "EXPERIMENT_TIME")
        require(self.registered_at < self.start < self.end, "PREREGISTRATION_TIME")
        require(self.metric in ("resolution_minutes", "nps_points"), "METRIC")
        token(self.cuped_version)
        integer(self.min_per_arm, "MIN_PER_ARM", 2, 5000)
        number(self.srm_alpha, "SRM_ALPHA", 0, 1)
        require(0 < self.srm_alpha < 1, "SRM_ALPHA")
        number(self.max_error_rate, "ERROR_LIMIT", 0, 1)

    @property
    def version(self):
        return digest(asdict(self))


def assign(spec, unit_id):
    require(unit_id in spec.cohort, "UNREGISTERED_UNIT")
    # Seed is a fixture fixed BEFORE outcomes. Never search seeds for a favorable result.
    key = f"{spec.experiment_id}\0{spec.seed}\0{unit_id}".encode()
    return "T" if int.from_bytes(hashlib.sha256(key).digest()[:8], "big") < 2 ** 63 else "C"


def srm_pvalue(treatment, total):
    """Exact two-sided binomial p for the only supported allocation, p=0.5."""
    integer(total, "SRM_TOTAL", 1, 10000)
    integer(treatment, "SRM_TREATMENT", 0, total)
    k = min(treatment, total - treatment)
    logs = [math.lgamma(total + 1) - math.lgamma(i + 1) - math.lgamma(total - i + 1)
            - total * math.log(2) for i in range(k + 1)]
    largest = max(logs)
    return min(1.0, 2 * math.exp(largest) * math.fsum(math.exp(v - largest) for v in logs))


def effect(control, treatment):
    require(len(control) >= 2 and len(treatment) >= 2, "EFFECT_TOO_SMALL")
    delta = fmean(treatment) - fmean(control)
    estimator_variance = variance(control) / len(control) + variance(treatment) / len(treatment)
    se = math.sqrt(estimator_variance)
    return {"control_mean": fmean(control), "treatment_mean": fmean(treatment), "delta_T_minus_C": delta,
            "variance_of_estimator": estimator_variance, "standard_error": se,
            "normal_approx_95_ci": [delta - 1.96 * se, delta + 1.96 * se] if se else None,
            "warning": None if se else "DEGENERATE_VARIANCE_NORMAL_CI_UNRELIABLE"}


def analyze(spec, cuped, rows, *, as_of):
    integer(as_of, "AS_OF")
    require(cuped.version == spec.cuped_version, "CUPED_VERSION")
    require(cuped.metric == spec.metric, "CUPED_METRIC")
    require(cuped.fitted_before <= spec.registered_at, "CUPED_NOT_FROZEN")
    base = {"project": "R3", "policy": "fixed-horizon-experiment-v1", "spec": asdict(spec),
            "spec_version": spec.version, "cuped": asdict(cuped), "as_of": as_of,
            "input_digest": digest(rows), "rollout_executed": False,
            "trace": ["verify_frozen_spec", "fixed_horizon"]}
    if as_of < spec.end:
        return {**base, "status": "NOT_READY", "reasons": ["FIXED_HORIZON_NOT_REACHED"], "effect": None}
    ids, counts, outcomes, adjusted, errors = set(), {"C": 0, "T": 0}, {"C": [], "T": []}, {"C": [], "T": []}, {"C": 0, "T": 0}
    for row in rows:
        fields(row, ("id", "arm", "assigned_at", "x_at", "x", "feature_version", "error", "outcome", "outcome_at"), "EXPERIMENT_FIELDS")
        unit = token(row["id"])
        require(unit not in ids, "DUPLICATE_UNIT")
        ids.add(unit)
        arm = row["arm"]
        require(arm == assign(spec, unit), "ASSIGNMENT_MISMATCH")
        integer(row["assigned_at"], "ASSIGNMENT_TIME")
        require(row["assigned_at"] == spec.start, "ASSIGNMENT_TIME")
        integer(row["x_at"], "COVARIATE_TIME")
        require(row["x_at"] < spec.start, "POST_TREATMENT_COVARIATE")
        require(row["feature_version"] == cuped.feature_version, "COVARIATE_VERSION")
        number(row["x"], "COVARIATE", -1000000, 1000000)
        require(type(row["error"]) is bool, "ERROR_FLAG")
        counts[arm] += 1
        errors[arm] += row["error"]
        if row["outcome"] is None:
            require(row["outcome_at"] is None, "MISSING_OUTCOME_TIME")
            continue
        y = number(row["outcome"], "OUTCOME", -1000000, 1000000)
        if spec.metric == "nps_points":
            require(y in (-100, 0, 100), "NPS_CONTRIBUTION")
        else:
            require(y >= 0, "NEGATIVE_DURATION")
        integer(row["outcome_at"], "OUTCOME_TIME")
        require(spec.start <= row["outcome_at"] <= min(spec.end, as_of), "OUTCOME_WINDOW")
        outcomes[arm].append(y)
        adjusted[arm].append(y - cuped.theta * (row["x"] - cuped.x_mean))
    reasons = []
    if ids != set(spec.cohort):
        reasons.append("MISSING_ASSIGNMENTS")
    pvalue = srm_pvalue(counts["T"], len(rows)) if rows else None
    if pvalue is not None and pvalue < spec.srm_alpha:
        reasons.append("SAMPLE_RATIO_MISMATCH")
    if min(counts.values()) < spec.min_per_arm:
        reasons.append("TOO_FEW_PER_ARM")
    if any(len(outcomes[a]) != counts[a] for a in counts):
        reasons.append("MISSING_OUTCOMES")
    rates = {a: errors[a] / counts[a] if counts[a] else None for a in counts}
    if any(v is not None and v > spec.max_error_rate for v in rates.values()):
        reasons.append("ERROR_GUARDRAIL")
    diagnostic = {"assignment_counts": counts, "outcome_counts": {a: len(v) for a, v in outcomes.items()},
                  "missing_assignment_n": len(spec.cohort) - len(ids), "srm_pvalue": pvalue, "error_rates": rates}
    base["trace"] += ["assignment_and_observation_integrity", "srm", "complete_outcomes", "guardrail"]
    if reasons:
        base["trace"].append("hold_without_estimation")
        return {**base, **diagnostic, "status": "HOLD", "reasons": reasons, "effect": None}
    base["trace"].append("estimate")
    raw, adj = effect(outcomes["C"], outcomes["T"]), effect(adjusted["C"], adjusted["T"])
    ratio = adj["variance_of_estimator"] / raw["variance_of_estimator"] if raw["variance_of_estimator"] else None
    return {**base, **diagnostic, "status": "READOUT_READY", "reasons": [],
            "effect": {"raw": raw, "cuped": adj, "estimated_variance_ratio": ratio},
            "interpretation": "synthetic fixed-horizon readout; interval is approximate; no business or rollout conclusion"}
