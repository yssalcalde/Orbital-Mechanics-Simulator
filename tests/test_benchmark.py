import csv
from pathlib import Path

import numpy as np

from analysis.benchmark_errors import (
    METHODS,
    load_system,
    observed_orders,
    simulate,
)

DATA_FILE = Path(__file__).parents[1] / "data" / "Moon-Earth.txt"


def test_simulation_reports_failed_method_explicitly() -> None:
    system = load_system(DATA_FILE)
    result = simulate("not-a-method", system, 1.0, 1.0)
    assert result.failure is not None
    assert result.positions.shape[0] == result.times.shape[0]


def test_observed_orders_calculate_expected_order_and_nan() -> None:
    orders = observed_orders([16.0, 4.0, float("nan")], (4.0, 2.0, 1.0))
    assert np.isnan(orders[0])
    assert orders[1] == 2.0
    assert np.isnan(orders[2])


def test_all_integrators_complete_short_simulation() -> None:
    system = load_system(DATA_FILE)
    for method in METHODS:
        result = simulate(method, system, 60.0, 600.0)
        assert result.failure is None, result.failure
        assert np.all(np.isfinite(result.positions))


def test_generated_summary_contains_order_and_failure_columns() -> None:
    summary = Path(__file__).parents[1] / "assets" / "benchmark_summary.csv"
    with summary.open(newline="", encoding="utf-8") as summary_file:
        columns = csv.DictReader(summary_file).fieldnames
    assert columns is not None
    assert {"observed_order", "status"} <= set(columns)
