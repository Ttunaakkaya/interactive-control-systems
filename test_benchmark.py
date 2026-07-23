"""Reproducibility and artifact-contract tests for benchmark.py."""

import csv
import json

from benchmark import (
    SCENARIOS,
    main,
    run_benchmark,
    run_benchmark_case,
    write_benchmark_report,
)


def test_same_case_and_seed_produce_identical_trajectory_and_metrics():
    first = run_benchmark_case(
        scenario=SCENARIOS["noisy_recovery"],
        controller_id="lqr",
        seed=17,
    )
    second = run_benchmark_case(
        scenario=SCENARIOS["noisy_recovery"],
        controller_id="lqr",
        seed=17,
    )

    assert first == second
    assert len(first.trajectory_sha256) == 64


def test_different_noise_seeds_change_the_trajectory():
    first = run_benchmark_case(
        scenario=SCENARIOS["noisy_recovery"],
        controller_id="lqr",
        seed=17,
    )
    second = run_benchmark_case(
        scenario=SCENARIOS["noisy_recovery"],
        controller_id="lqr",
        seed=18,
    )

    assert first.trajectory_sha256 != second.trajectory_sha256


def test_suite_normalizes_order_and_is_reproducible():
    first = run_benchmark(
        suite="quick",
        controller_ids=("lqi", "lqr"),
        seeds=(9, 3, 9),
    )
    second = run_benchmark(
        suite="quick",
        controller_ids=("lqr", "lqi"),
        seeds=(3, 9),
    )

    assert first.controller_ids == ("lqr", "lqi")
    assert first.seeds == (3, 9)
    assert first.protocol_sha256 == second.protocol_sha256
    assert first.results_sha256 == second.results_sha256
    assert first.to_json() == second.to_json()
    assert len(first.runs) == 8
    assert len(first.aggregates) == 4


def test_json_and_csv_artifacts_are_stable(tmp_path):
    report = run_benchmark(
        suite="quick",
        controller_ids=("lqr",),
        seeds=(5,),
    )
    json_path, csv_path = write_benchmark_report(report, tmp_path)
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()

    write_benchmark_report(report, tmp_path)

    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
    parsed = json.loads(first_json)
    assert parsed["implementation_sha256"] == report.implementation_sha256
    assert parsed["protocol_sha256"] == report.protocol_sha256
    assert parsed["results_sha256"] == report.results_sha256
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["scenario_id"] for row in rows} == {
        "nominal_recovery",
        "noisy_recovery",
    }


def test_cli_writes_requested_quick_report(tmp_path):
    exit_code = main(
        [
            "--suite",
            "quick",
            "--controllers",
            "lqr",
            "--seeds",
            "23",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "benchmark_results.json").is_file()
    assert (tmp_path / "benchmark_results.csv").is_file()
