"""Hand-designed, public synthetic fixtures; no imported customer data."""
from . import conformal, experiment

LABELS = ("complaint", "praise", "other")


def prediction(unit, values, *, channel="mobile", observed_at=20, **changes):
    return {"id": unit, "channel": channel, "observed_at": observed_at,
            "model_version": "synthetic-scores-v1", "taxonomy_version": "voc-v1",
            "probabilities": dict(zip(LABELS, values)) if values is not None else None,
            "eligible": True, "risk_flag": False, "drift_flag": False, **changes}


def conformal_case():
    calibration = []
    for i, score in enumerate([.1 + .03 * i for i in range(18)] + [.9]):
        row = prediction(f"CAL-{i:02}", (1 - score, score / 2, score / 2), observed_at=10)
        row["truth"] = "complaint"
        calibration.append(row)
    model = conformal.fit(calibration, labels=LABELS, model_version="synthetic-scores-v1",
                          taxonomy_version="voc-v1", cutoff=10, training_ids=("TRAIN-1",))
    rows = [prediction("SYN-1", (.8, .1, .1)), prediction("SYN-2", (.1, .8, .1)),
            prediction("SYN-3", (.45, .4, .15)), prediction("SYN-4", (.34, .33, .33)),
            prediction("SYN-5", (.1, .8, .1), risk_flag=True),
            prediction("SYN-6", (.8, .1, .1), channel="branch"),
            prediction("SYN-7", (.8, .1, .1), model_version="new-model-v2"),
            prediction("SYN-8", None)]
    truths = dict(zip((r["id"] for r in rows), ("complaint", "complaint", "praise", "other", "praise", "complaint", "complaint", "other")))
    return model, rows, truths


def cell(promoter, passive, detractor, *, population=1000, invited=100):
    return dict(population=population, invited=invited, promoter=promoter, passive=passive, detractor=detractor)


def nps_case():
    before = {"high": cell(14, 4, 2), "low": cell(24, 16, 40)}
    after = {"high": cell(56, 16, 8), "low": cell(6, 4, 10)}
    return before, after, {"high": 1000, "low": 1000}


def experiment_case():
    history = [{"id": f"HIST-{i}", "x": 30 + i, "y": 10 + .8 * (30 + i) + ((i % 5) - 2),
                "x_at": 1, "y_at": 3} for i in range(60)]
    cuped = experiment.fit_cuped(history, fitted_before=8)
    spec = experiment.Experiment("cx-resolution-v1", "fixed-before-results-v1", tuple(f"SYN-{i:04}" for i in range(400)),
                                 registered_at=9, start=10, end=20, cuped_version=cuped.version)
    rows = []
    for i, unit in enumerate(spec.cohort):
        arm = experiment.assign(spec, unit)
        x = 30 + (i * 17 % 61)
        # Synthetic structural effect -4 minutes; no empirical improvement is claimed.
        y = 10 + .8 * x + (i % 7 - 3) - (4 if arm == "T" else 0)
        rows.append({"id": unit, "arm": arm, "assigned_at": 10, "x": x, "x_at": 5,
                     "feature_version": cuped.feature_version,
                     "outcome": y, "outcome_at": 19, "error": i % 100 == 0})
    return spec, cuped, rows
