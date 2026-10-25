"""Tkinter controls for selecting an orbital simulation."""

from __future__ import annotations

import csv
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import DoubleVar, StringVar, filedialog, messagebox

from .physics import Planet
from .visualizer import visualize_system

METHODS = {
    "Explicit Euler": "euler",
    "Implicit Euler": "euler_implicit",
    "RK4": "rk4",
    "Velocity Verlet": "verlet",
}


@dataclass
class OrbitalSystem:
    central_name: str
    central_mass: float
    central_color: str
    central_size: int
    planets: list[Planet]


def load_system(path: str | Path) -> OrbitalSystem:
    """Load one central body and orbiting bodies from a CSV text file."""
    with Path(path).open(newline="", encoding="utf-8") as data_file:
        rows = list(csv.DictReader(data_file))

    central_rows = [row for row in rows if row["type"].strip().lower() == "central"]
    body_rows = [row for row in rows if row["type"].strip().lower() == "body"]
    if len(central_rows) != 1:
        raise ValueError("The file must contain exactly one central body.")
    if not body_rows:
        raise ValueError("The file must contain at least one orbiting body.")

    central = central_rows[0]
    planets = [
        Planet(
            name=row["name"],
            mass=float(row["mass_kg"]),
            distance=float(row["distance_m"]),
            velocity=float(row["velocity_m_s"]),
            inclination=float(row["inclination_deg"]),
            color=row["color"],
            size=int(row["size"]),
        )
        for row in body_rows
    ]
    return OrbitalSystem(
        central_name=central["name"],
        central_mass=float(central["mass_kg"]),
        central_color=central["color"],
        central_size=int(central["size"]),
        planets=planets,
    )


def create_app(initial_file: str | Path | None = None) -> tk.Tk:
    """Create the simulation setup window."""
    root = tk.Tk()
    root.title("Orbital Mechanics Simulator")
    root.resizable(False, False)

    selected_file = StringVar(
        root,
        str(initial_file) if initial_file else str(Path("data") / "Moon-Earth.txt"),
    )
    selected_method = StringVar(root, "Velocity Verlet")
    time_step = DoubleVar(root, 7200.0)

    def choose_file() -> None:
        path = filedialog.askopenfilename(
            title="Select orbital system",
            filetypes=[("Orbital data", "*.txt"), ("All files", "*.*")],
            initialdir=str(Path("data")),
        )
        if path:
            selected_file.set(path)

    def start() -> None:
        try:
            system = load_system(selected_file.get())
            dt = time_step.get()
            if dt <= 0:
                raise ValueError("The time step must be positive.")
        except (OSError, KeyError, TypeError, ValueError) as error:
            messagebox.showerror("Invalid simulation setup", str(error), parent=root)
            return

        root.withdraw()
        visualize_system(
            system,
            method=METHODS[selected_method.get()],
            dt=dt,
            on_close=root.deiconify,
        )

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack()

    tk.Label(frame, text="System file").grid(row=0, column=0, sticky="w", pady=4)
    tk.Entry(frame, textvariable=selected_file, width=52).grid(
        row=0, column=1, padx=6, pady=4
    )
    tk.Button(frame, text="Browse...", command=choose_file).grid(
        row=0, column=2, pady=4
    )

    tk.Label(frame, text="Integration method").grid(
        row=1, column=0, sticky="w", pady=4
    )
    method_menu = tk.OptionMenu(frame, selected_method, *METHODS)
    method_menu.config(width=20)
    method_menu.grid(row=1, column=1, sticky="w", padx=6, pady=4)

    tk.Label(frame, text="Time step (seconds)").grid(
        row=2, column=0, sticky="w", pady=4
    )
    tk.Entry(frame, textvariable=time_step, width=12).grid(
        row=2, column=1, sticky="w", padx=6, pady=4
    )

    tk.Button(frame, text="Start simulation", command=start).grid(
        row=3, column=0, columnspan=3, pady=(12, 0)
    )
    return root


def main() -> None:
    app = create_app()
    app.mainloop()


if __name__ == "__main__":
    main()