from analysis.benchmark_errors import load_system, simulate


def test_simulation_reports_failed_method_explicitly() -> None:
    system = load_system(
        __import__("pathlib").Path(__file__).parents[1] / "data" / "Moon-Earth.txt"
    )
    result = simulate("not-a-method", system, 1.0, 1.0)
    assert result.failure is not None
    assert result.positions.shape[0] == result.times.shape[0]
