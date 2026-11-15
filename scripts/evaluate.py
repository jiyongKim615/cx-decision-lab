"""Evaluate fixed synthetic scenarios; write measured values, not performance claims."""
import json
from math import isclose
from pathlib import Path

from cx_research_lab import experiment
from cx_research_lab.demo import run
from cx_research_lab.fixtures import experiment_case


def evaluate():
    result = run()
    classification = result["R1"]["metrics"]
    assert (classification["total"], classification["auto_n"], classification["review_n"]) == (8, 2, 6)
    assert classification["auto_error_n"] == 1
    manual = run(manual_only=True)["R1"]["metrics"]
    assert (manual["auto_n"], manual["review_n"]) == (0, 8)
    nps = result["R2"]
    parts = nps["decomposition"]
    assert isclose(parts["raw_delta"], parts["within"] + parts["mix"])
    assert (parts["raw_delta"], parts["within"], nps["standardized_delta"]) == (48, 0, 0)
    assert result["R2_missing_stratum"]["status"] == "HOLD"
    assert result["R2_missing_stratum"]["standardized_delta"] is None
    normal = result["R3"]
    assert normal["status"] == "READOUT_READY"
    for key in ("R3_missing_outcome", "R3_lost_exposures"):
        assert result[key]["status"] == "HOLD" and result[key]["effect"] is None
    assert "SAMPLE_RATIO_MISMATCH" in result["R3_lost_exposures"]["reasons"]
    spec, cuped, rows = experiment_case()
    early = experiment.analyze(spec, cuped, rows, as_of=19)
    assert early["status"] == "NOT_READY" and early["effect"] is None
    return {
        "data_origin": "hand-authored synthetic fixtures",
        "network_calls": 0,
        "classification": classification,
        "manual_mode": {"auto_n": manual["auto_n"], "review_n": manual["review_n"]},
        "nps": {"raw_before": nps["before"]["raw_nps"], "raw_after": nps["after"]["raw_nps"],
                "raw_delta": parts["raw_delta"], "within": parts["within"], "mix": round(parts["mix"], 10),
                "standardized_delta": nps["standardized_delta"], "missing_stratum_status": "HOLD"},
        "experiment": {"status": normal["status"], "assignment_counts": normal["assignment_counts"],
                       "effect": normal["effect"], "early_status": early["status"],
                       "missing_outcome_reasons": result["R3_missing_outcome"]["reasons"],
                       "lost_exposures_reasons": result["R3_lost_exposures"]["reasons"]},
    }


def main():
    result = evaluate()
    path = Path("artifacts/evaluation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Scenario checks passed: {path}")


if __name__ == "__main__":
    main()
