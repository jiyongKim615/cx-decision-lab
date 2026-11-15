"""Independent arithmetic oracles, adverse inputs and safety invariants for R1-R3."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cx_research_lab import conformal, experiment, nps
from cx_research_lab.common import ContractError, digest, persist
from cx_research_lab.demo import run
from cx_research_lab.fixtures import LABELS, cell, conformal_case, experiment_case, nps_case, prediction


class ConformalTests(unittest.TestCase):
    def setUp(self):
        self.model, self.rows, self.truths = conformal_case()

    def test_quantile_hand_order_statistic(self):
        self.assertEqual(conformal.quantile([.1, .4, .2, .3], .4), (.3, 3))

    def test_quantile_finite_sample_insufficient_alpha(self):
        self.assertEqual(conformal.quantile([.1, .2], .1), (1.0, 3))

    def test_exchangeable_leave_one_out_coverage_oracle(self):
        scores = [.05, .15, .25, .35, .45]
        covered = sum(s <= conformal.quantile(scores[:i] + scores[i + 1:], .4)[0] for i, s in enumerate(scores))
        self.assertEqual(covered, 3)  # Exactly 3/5, all held-out ranks equally likely.

    def test_quantile_rejects_invalid_values(self):
        for values, alpha in [([], .1), ([float("nan")], .1), ([.1], 0), ([.1], 1), ([.1], True)]:
            with self.subTest(values=values, alpha=alpha), self.assertRaises(ContractError):
                conformal.quantile(values, alpha)

    def test_binary_sets_are_supported(self):
        rows = [dict(prediction("CAL-1", (.8, .2, 0), observed_at=1), truth="complaint")]
        rows[0]["probabilities"].pop("other")
        fitted = conformal.fit(rows, labels=("complaint", "praise"), model_version="synthetic-scores-v1",
                               taxonomy_version="voc-v1", cutoff=1, alpha=.5, min_group=1)
        row = prediction("SYN-1", (.8, .2, 0), observed_at=2)
        row["probabilities"].pop("other")
        self.assertEqual(conformal.route(fitted, row)["prediction_set"], ["complaint"])

    def test_set_boundary_included(self):
        row = prediction("BOUNDARY", (.39, .38, .23))
        self.assertEqual(conformal.route(self.model, row)["prediction_set"], ["complaint"])

    def test_singleton_is_not_accuracy_guarantee(self):
        report = conformal.run(self.model, self.rows, self.truths)
        self.assertEqual(report["metrics"]["auto_n"], 2)
        self.assertEqual(report["metrics"]["auto_error_n"], 1)
        self.assertEqual(report["metrics"]["auto_error_rate"], .5)

    def test_set_coverage_excludes_unavailable_calibration(self):
        m = conformal.run(self.model, self.rows, self.truths)["metrics"]
        self.assertEqual((m["total"], m["set_coverage_n"], m["set_covered"]), (8, 5, 3))

    def test_multi_and_empty_sets_require_review(self):
        for i, reason in [(2, "MULTIPLE_LABELS"), (3, "EMPTY_SET")]:
            decision = conformal.route(self.model, self.rows[i])
            self.assertEqual(decision["decision"], "REVIEW")
            self.assertIn(reason, decision["reasons"])

    def test_hard_gates_survive_confident_singleton(self):
        for change, reason in [({"risk_flag": True}, "RISK_REVIEW"), ({"eligible": False}, "INELIGIBLE_FOR_AUTO"),
                               ({"drift_flag": True}, "DECLARED_DRIFT"), ({"model_version": "v2"}, "VERSION_MISMATCH"),
                               ({"taxonomy_version": "v2"}, "VERSION_MISMATCH")]:
            with self.subTest(change=change):
                decision = conformal.route(self.model, {**self.rows[0], **change})
                self.assertEqual(decision["decision"], "REVIEW")
                self.assertIn(reason, decision["reasons"])

    def test_capacity_does_not_auto_accept_overflow(self):
        r = conformal.run(self.model, self.rows, self.truths, review_capacity=1)
        self.assertEqual(r["review_now"], ["SYN-5"])
        self.assertEqual(len(r["review_backlog"]), 5)
        self.assertEqual(r["metrics"]["auto_n"], 2)
        self.assertTrue(all(not d["executed_customer_action"] for d in r["decisions"]))

    def test_manual_only_rollback(self):
        r = conformal.run(self.model, self.rows, self.truths, manual_only=True)
        self.assertEqual(r["metrics"]["auto_n"], 0)
        self.assertEqual(r["metrics"]["review_n"], 8)
        self.assertIsNone(r["metrics"]["auto_error_rate"])

    def test_unknown_channel_and_missing_scores_review(self):
        self.assertIn("INSUFFICIENT_CHANNEL_CALIBRATION", conformal.route(self.model, self.rows[5])["reasons"])
        self.assertIn("NO_SCORES", conformal.route(self.model, self.rows[7])["reasons"])

    def test_small_known_channel_review(self):
        model = replace(self.model, min_group=20)
        self.assertIn("INSUFFICIENT_CHANNEL_CALIBRATION", conformal.route(model, self.rows[0])["reasons"])

    def test_probabilities_reject_nan_bool_sum_and_label(self):
        for values in [{"complaint": float("nan"), "praise": .1, "other": .1},
                       {"complaint": True, "praise": 0, "other": 0},
                       {"complaint": .4, "praise": .1, "other": .1}, {"wrong": 1}]:
            with self.subTest(values=values), self.assertRaises(ContractError):
                conformal.route(self.model, {**self.rows[0], "probabilities": values})

    def test_fit_rejects_overlap_duplicate_future_and_version(self):
        good = dict(prediction("CAL-X", (.8, .1, .1), observed_at=1), truth="complaint")
        kwargs = dict(labels=LABELS, model_version="synthetic-scores-v1", taxonomy_version="voc-v1", cutoff=1)
        for rows, extra in [([good], {"training_ids": ["CAL-X"]}), ([good, good], {}),
                            ([{**good, "observed_at": 2}], {}), ([{**good, "model_version": "v2"}], {})]:
            with self.subTest(rows=rows), self.assertRaises(ContractError):
                conformal.fit(rows, **kwargs, **extra)

    def test_runtime_overlap_and_cutoff(self):
        for changes in [{"id": "TRAIN-1"}, {"id": "CAL-01"}, {"observed_at": 10}]:
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                conformal.route(self.model, {**self.rows[0], **changes})

    def test_duplicate_batch_and_truth_mismatch(self):
        with self.assertRaises(ContractError):
            conformal.run(self.model, self.rows + [self.rows[0]], self.truths)
        with self.assertRaises(ContractError):
            conformal.run(self.model, self.rows, {})


class NpsTests(unittest.TestCase):
    def setUp(self):
        self.before, self.after, self.target = nps_case()

    def test_simpson_fixture_hand_calculation(self):
        r = nps.compare(self.before, self.after, self.target)
        self.assertEqual((r["before"]["raw_nps"], r["after"]["raw_nps"]), (-4, 44))
        self.assertAlmostEqual(r["decomposition"]["mix"], 48)
        self.assertEqual(r["decomposition"]["within"], 0)
        self.assertEqual(r["standardized_delta"], 0)

    def test_fixed_composition_measures_within_change(self):
        after = {"high": cell(16, 2, 2), "low": cell(32, 16, 32)}
        r = nps.compare(self.before, after, self.target)
        self.assertEqual(r["decomposition"]["mix"], 0)
        self.assertAlmostEqual(r["decomposition"]["within"], 18)
        self.assertEqual(r["standardized_delta"], 15)

    def test_decomposition_identity_varied_tables(self):
        # Independent brute-force totals for 30 different changing distributions.
        for i in range(1, 31):
            before = {"a": cell(i, 2, 3), "b": cell(4, 1, i + 2)}
            after = {"a": cell(2, i + 1, 4), "b": cell(i + 3, 1, 2)}
            r = nps.compare(before, after, {"a": 50, "b": 50})
            def raw(c):
                return 100 * sum(v["promoter"] - v["detractor"] for v in c.values()) / sum(v["promoter"] + v["passive"] + v["detractor"] for v in c.values())
            self.assertAlmostEqual(r["decomposition"]["within"] + r["decomposition"]["mix"], raw(after) - raw(before))

    def test_missing_stratum_holds_without_renormalizing(self):
        self.after.pop("low")
        r = nps.compare(self.before, self.after, self.target)
        self.assertEqual(r["status"], "HOLD")
        self.assertIsNone(r["after"]["standardized_nps"])
        self.assertIsNone(r["decomposition"])
        self.assertEqual(r["after"]["missing_target_strata"], ["low"])

    def test_zero_responses_not_zero_nps(self):
        empty = {"high": cell(0, 0, 0), "low": cell(0, 0, 0)}
        r = nps.compare(self.before, empty, self.target)
        self.assertIsNone(r["after"]["raw_nps"])
        self.assertIsNone(r["after"]["standardized_nps"])
        self.assertEqual(r["status"], "HOLD")

    def test_kish_ess_independent_individual_weights(self):
        r = nps.compare(self.before, self.after, self.target)
        weights = [.5 / 20] * 20 + [.5 / 80] * 80
        ess = sum(weights) ** 2 / sum(w * w for w in weights)
        self.assertAlmostEqual(r["before"]["weight_only_ess"], ess)
        self.assertAlmostEqual(ess, 64)

    def test_low_response_warns_even_if_point_estimate_exists(self):
        r = nps.compare(self.before, self.after, self.target, min_response_rate=.9)
        self.assertIn("high:LOW_RESPONSE", r["before"]["warnings"])
        self.assertFalse(r["causal_claim_allowed"])

    def test_counts_reject_invalid_type_and_total(self):
        for value in [True, -1, .5, float("nan"), 101]:
            before = deepcopy(self.before)
            before["high"]["promoter"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                nps.compare(before, self.after, self.target)

    def test_target_unknown_and_zero_rejected(self):
        for target in [{"high": 0, "low": 0}, {"high": 1}, {"high": True, "low": 10}]:
            with self.subTest(target=target), self.assertRaises(ContractError):
                nps.compare(self.before, self.after, target)

    def test_reference_change_changes_evidence(self):
        first = nps.compare(self.before, self.after, self.target)
        second = nps.compare(self.before, self.after, {"high": 3, "low": 1})
        self.assertNotEqual(first["reference_digest"], second["reference_digest"])
        self.assertEqual(second["before"]["standardized_nps"], 40)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.cuped, self.rows = experiment_case()

    def analyze(self, rows=None, **kwargs):
        return experiment.analyze(self.spec, self.cuped, self.rows if rows is None else rows, as_of=kwargs.get("as_of", 20))

    def test_assignment_stable_and_order_independent(self):
        first = {x: experiment.assign(self.spec, x) for x in self.spec.cohort}
        self.assertEqual(first, {x: experiment.assign(self.spec, x) for x in reversed(self.spec.cohort)})
        self.assertEqual(self.analyze()["effect"], self.analyze(list(reversed(self.rows)))["effect"])

    def test_spec_frozen_and_version_binds_rules(self):
        with self.assertRaises(FrozenInstanceError):
            self.spec.seed = "changed"
        self.assertNotEqual(replace(self.spec, max_error_rate=.1).version, self.spec.version)

    def test_srm_exact_hand_probabilities(self):
        # 2*(choose(10,0)+choose(10,1))/2^10.
        self.assertAlmostEqual(experiment.srm_pvalue(1, 10), 22 / 1024)
        self.assertAlmostEqual(experiment.srm_pvalue(0, 10), 2 / 1024)
        self.assertEqual(experiment.srm_pvalue(5, 10), 1)
        self.assertAlmostEqual(experiment.srm_pvalue(9, 10), 22 / 1024)

    def test_srm_large_extreme_is_finite(self):
        self.assertLess(experiment.srm_pvalue(0, 10000), 1e-100)
        self.assertEqual(experiment.srm_pvalue(5000, 10000), 1)

    def test_cuped_hand_slope_and_constant_covariate(self):
        history = [{"id": f"H-{i}", "x": i, "y": 2 * i + 5, "x_at": 1, "y_at": 2} for i in range(4)]
        fitted = experiment.fit_cuped(history, fitted_before=3)
        self.assertEqual((fitted.theta, fitted.x_mean), (2, 1.5))
        for row in history:
            row["x"] = 1
        self.assertEqual(experiment.fit_cuped(history, fitted_before=3).theta, 0)

    def test_cuped_history_after_cutoff_rejected(self):
        history = [{"id": f"H-{i}", "x": i, "y": i, "x_at": 1, "y_at": 3} for i in range(2)]
        with self.assertRaisesRegex(ContractError, "HISTORICAL_LEAKAGE"):
            experiment.fit_cuped(history, fitted_before=3)

    def test_effect_hand_difference_and_variance(self):
        r = experiment.effect([1, 3, 5], [3, 5, 7])
        self.assertEqual(r["delta_T_minus_C"], 2)
        self.assertAlmostEqual(r["variance_of_estimator"], 8 / 3)

    def test_readout_is_synthetic_not_rollout(self):
        r = self.analyze()
        self.assertEqual(r["status"], "READOUT_READY")
        self.assertEqual(sum(r["assignment_counts"].values()), 400)
        self.assertFalse(r["rollout_executed"])
        self.assertLess(r["effect"]["estimated_variance_ratio"], 1)

    def test_early_peek_not_ready(self):
        r = self.analyze(as_of=19)
        self.assertEqual(r["status"], "NOT_READY")
        self.assertIsNone(r["effect"])
        self.assertNotIn("estimate", r["trace"])

    def test_missing_outcome_holds_not_zero_imputation(self):
        self.rows[0].update(outcome=None, outcome_at=None)
        r = self.analyze()
        self.assertIn("MISSING_OUTCOMES", r["reasons"])
        self.assertEqual(sum(r["assignment_counts"].values()), 400)
        self.assertEqual(sum(r["outcome_counts"].values()), 399)
        self.assertIsNone(r["effect"])
        self.assertNotIn("estimate", r["trace"])

    def test_lost_treatment_exposures_srm_and_cohort_gate(self):
        rows = [r for i, r in enumerate(self.rows) if r["arm"] == "C" or i % 4 == 0]
        r = self.analyze(rows)
        self.assertIn("SAMPLE_RATIO_MISMATCH", r["reasons"])
        self.assertIn("MISSING_ASSIGNMENTS", r["reasons"])
        self.assertIsNone(r["effect"])

    def test_balanced_event_loss_still_fails_cohort_check(self):
        dropped = {next(r["id"] for r in self.rows if r["arm"] == arm) for arm in ["C", "T"]}
        r = self.analyze([row for row in self.rows if row["id"] not in dropped])
        self.assertIn("MISSING_ASSIGNMENTS", r["reasons"])

    def test_guardrail_overrides_good_effect(self):
        for row in self.rows:
            row["error"] = row["arm"] == "T"
        r = self.analyze()
        self.assertIn("ERROR_GUARDRAIL", r["reasons"])
        self.assertIsNone(r["effect"])

    def test_duplicate_unit_rejected(self):
        with self.assertRaisesRegex(ContractError, "DUPLICATE_UNIT"):
            self.analyze(self.rows + [self.rows[0]])

    def test_assignment_tampering_rejected(self):
        self.rows[0]["arm"] = "C" if self.rows[0]["arm"] == "T" else "T"
        with self.assertRaisesRegex(ContractError, "ASSIGNMENT_MISMATCH"):
            self.analyze()

    def test_post_treatment_covariate_rejected(self):
        self.rows[0]["x_at"] = 10
        with self.assertRaisesRegex(ContractError, "POST_TREATMENT_COVARIATE"):
            self.analyze()

    def test_outcome_window_rejected(self):
        self.rows[0]["outcome_at"] = 21
        with self.assertRaisesRegex(ContractError, "OUTCOME_WINDOW"):
            self.analyze()

    def test_version_mismatch_rejected(self):
        with self.assertRaisesRegex(ContractError, "CUPED_VERSION"):
            experiment.analyze(self.spec, replace(self.cuped, theta=99), self.rows, as_of=20)

    def test_nan_and_bool_rejected(self):
        for key, value in [("outcome", float("nan")), ("x", True), ("error", 1), ("outcome", -1)]:
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                self.analyze(rows)

    def test_nps_is_unit_contribution_not_zero_to_ten_score(self):
        history = [{"id": f"N-{i}", "x": i, "y": 100, "x_at": 1, "y_at": 3} for i in range(4)]
        cuped = experiment.fit_cuped(history, fitted_before=8, metric="nps_points", feature_version="previous_nps-v1")
        spec = replace(self.spec, metric="nps_points", cuped_version=cuped.version)
        rows = deepcopy(self.rows)
        for row in rows:
            row["outcome"] = 100
            row["feature_version"] = cuped.feature_version
        result = experiment.analyze(spec, cuped, rows, as_of=20)
        self.assertEqual(result["status"], "READOUT_READY")
        self.assertIsNone(result["effect"]["raw"]["normal_approx_95_ci"])
        rows[0]["outcome"] = 9
        with self.assertRaisesRegex(ContractError, "NPS_CONTRIBUTION"):
            experiment.analyze(spec, cuped, rows, as_of=20)

    def test_covariate_model_cannot_silently_change_metric(self):
        with self.assertRaisesRegex(ContractError, "CUPED_METRIC"):
            experiment.analyze(replace(self.spec, metric="nps_points"), self.cuped, self.rows, as_of=20)

    def test_partial_record_has_contract_error(self):
        with self.assertRaisesRegex(ContractError, "EXPERIMENT_FIELDS"):
            self.analyze([{"id": "SYN-0000"}])

    def test_feature_version_change_rejected(self):
        self.rows[0]["feature_version"] = "changed-feature-v2"
        with self.assertRaisesRegex(ContractError, "COVARIATE_VERSION"):
            self.analyze()

    def test_bad_historical_coefficient_may_increase_variance(self):
        # Do not clamp ratios to <=1 or assume every historical predictor helps.
        cuped = replace(self.cuped, theta=-10)
        spec = replace(self.spec, cuped_version=cuped.version)
        r = experiment.analyze(spec, cuped, self.rows, as_of=20)
        self.assertGreater(r["effect"]["estimated_variance_ratio"], 1)


class ArtifactAndCliTests(unittest.TestCase):
    def test_deterministic_full_result(self):
        self.assertEqual(digest(run()), digest(run()))

    def test_atomic_idempotent_publication_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            data = {"fixture": 1}
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: persist(directory, data), range(4)))
            self.assertEqual(sum(not r["reused"] for r in results), 1)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            path = Path(results[0]["path"])
            path.write_text("corruption")
            with self.assertRaisesRegex(ContractError, "ARTIFACT_CONFLICT"):
                persist(directory, data)

    def test_distinct_versions_preserve_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            first = persist(directory, {"version": 1})
            second = persist(directory, {"version": 2})
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual(json.loads(Path(first["path"]).read_text()), {"version": 1})

    def test_cli_json_and_manual_rollback(self):
        r = subprocess.run([sys.executable, "-m", "cx_research_lab", "--project", "R1", "--manual-only"],
                           capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(r.stdout)["metrics"]["auto_n"], 0)

    def test_cli_invalid_alpha_has_clear_error(self):
        r = subprocess.run([sys.executable, "-m", "cx_research_lab", "--alpha", "0"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("CONTRACT_ERROR: ALPHA", r.stderr)


if __name__ == "__main__":
    unittest.main()
