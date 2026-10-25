from pathlib import Path

import pytest

from analysis.benchmark_errors import load_system as load_benchmark_system
from src.gui import load_system


DATA_FILE = Path(__file__).parents[1] / "data" / "Moon-Earth.txt"


def test_gui_parser_loads_example_system() -> None:
    system = load_system(DATA_FILE)
    assert system.central_name == "Earth"
    assert system.central_mass > 0
    assert len(system.planets) == 1


def test_benchmark_parser_loads_example_system() -> None:
    system = load_benchmark_system(DATA_FILE)
    assert system.central_mass > 0
    assert system.planet.name == "Moon"


def test_gui_parser_rejects_missing_orbiting_body(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(
        "type,name,mass_kg,distance_m,velocity_m_s,inclination_deg,color,size\n"
        "central,Earth,5.972e24,0,0,0,blue,15\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one"):
        load_system(path)
