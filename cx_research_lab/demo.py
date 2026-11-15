"""Every project includes a passing fixture and a deliberately adverse case."""
from copy import deepcopy

from . import conformal, experiment, nps
from .fixtures import conformal_case, experiment_case, nps_case


def run(*, alpha=.1, manual_only=False, review_capacity=3, drop_outcome=False):
    model, predictions, truths = conformal_case()
    if alpha != model.alpha:
        # Reconstruct quantiles from the documented calibration fixture, never edit q directly.
        from .fixtures import LABELS, prediction
        rows = [dict(prediction(f"CAL-{i:02}", (1 - s, s / 2, s / 2), observed_at=10), truth="complaint")
                for i, s in enumerate([.1 + .03 * j for j in range(18)] + [.9])]
        model = conformal.fit(rows, labels=LABELS, model_version=model.model_version,
                              taxonomy_version=model.taxonomy_version, cutoff=10, training_ids=model.training_ids, alpha=alpha)
    before, after, target = nps_case()
    empty = deepcopy(after)
    for key in ("promoter", "passive", "detractor"):
        empty["low"][key] = 0
    spec, cuped, rows = experiment_case()
    if drop_outcome:
        rows[0].update(outcome=None, outcome_at=None)
    missing = deepcopy(rows)
    missing[0].update(outcome=None, outcome_at=None)
    # A dropped treatment exposure simulates a biased event stream; the frozen cohort also detects loss.
    biased = [r for i, r in enumerate(rows) if r["arm"] == "C" or i % 4 == 0]
    return {"schema": "cx-research-1", "data_origin": "synthetic", "network_calls": 0,
            "R1": conformal.run(model, predictions, truths, review_capacity=review_capacity, manual_only=manual_only),
            "R2": nps.compare(before, after, target),
            "R2_missing_stratum": nps.compare(before, empty, target),
            "R3": experiment.analyze(spec, cuped, rows, as_of=20),
            "R3_missing_outcome": experiment.analyze(spec, cuped, missing, as_of=20),
            "R3_lost_exposures": experiment.analyze(spec, cuped, biased, as_of=20)}
