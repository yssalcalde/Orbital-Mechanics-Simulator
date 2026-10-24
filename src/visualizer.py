"""Matplotlib visualizer for the orbital simulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np

if TYPE_CHECKING:
    from .gui import OrbitalSystem


def visualize_system(
    system: OrbitalSystem,
    method: str,
    dt: float,
    on_close: Callable[[], None] | None = None,
) -> None:
    """Animate an orbital system with the selected integration method."""
    initial_states = [
        (planet.position.copy(), planet.velocity.copy()) for planet in system.planets
    ]
    central_position = np.zeros(3, dtype=np.float64)
    frame_steps = [1]
    time_step = [float(dt)]
    zoom = [1.0]
    paused = [False]

    figure = plt.figure(figsize=(11, 8), facecolor="black")
    axis = figure.add_subplot(111, projection="3d")
    figure.canvas.manager.set_window_title("Orbital Mechanics Simulator")
    axis.set_facecolor("black")

    all_distances = [np.linalg.norm(planet.position) for planet in system.planets]
    base_view_radius = max(all_distances, default=1.0) * 1.25
    stars = np.random.default_rng(7).uniform(
        -base_view_radius * 2, base_view_radius * 2, (250, 3)
    )

    def configure_axis() -> None:
        view_radius = base_view_radius / zoom[0]
        axis.set_xlim(-view_radius, view_radius)
        axis.set_ylim(-view_radius, view_radius)
        axis.set_zlim(-view_radius, view_radius)
        axis.set_axis_off()
        axis.grid(False)
        axis.set_box_aspect((1, 1, 1))

    def reset() -> None:
        for planet, (position, velocity) in zip(system.planets, initial_states):
            planet.position = position.copy()
            planet.velocity = velocity.copy()
            planet.trajectory = [planet.position.copy()]
        paused[0] = False

    def update(_frame: int) -> None:
        if not paused[0]:
            try:
                for _ in range(frame_steps[0]):
                    for planet in system.planets:
                        planet.update(
                            time_step[0],
                            system.central_mass,
                            central_position,
                            method,
                        )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                paused[0] = True
                axis.set_title(f"{method}: stopped - {error}")

        axis.clear()
        configure_axis()
        axis.scatter(stars[:, 0], stars[:, 1], stars[:, 2], color="white", s=2, alpha=0.8)
        axis.scatter(
            0,
            0,
            0,
            color=system.central_color,
            s=system.central_size**2,
            label=system.central_name,
        )
        for planet in system.planets:
            trajectory = np.asarray(planet.trajectory)
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=planet.color,
                linewidth=1,
            )
            axis.scatter(
                *planet.position,
                color=planet.color,
                s=planet.size**2,
                label=planet.name,
            )
        if not paused[0]:
            axis.set_title(
                f"{method} | dt = {time_step[0]:g} s | zoom = {zoom[0]:.1f}x",
                color="white",
            )
        legend = axis.legend(loc="upper left", fontsize=8)
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
        for text in legend.get_texts():
            text.set_color("white")

    def toggle_pause(_event) -> None:
        paused[0] = not paused[0]

    def reset_simulation(_event) -> None:
        reset()
        paused[0] = False

    def zoom_with_mouse(event) -> None:
        if event.button == "up":
            zoom[0] = min(zoom[0] * 1.15, 20.0)
        elif event.button == "down":
            zoom[0] = max(zoom[0] / 1.15, 0.1)

    figure.subplots_adjust(bottom=0.25)
    pause_axis = figure.add_axes((0.70, 0.05, 0.10, 0.05), facecolor="black")
    reset_axis = figure.add_axes((0.81, 0.05, 0.10, 0.05), facecolor="black")
    steps_axis = figure.add_axes((0.15, 0.14, 0.40, 0.03), facecolor="dimgray")
    dt_axis = figure.add_axes((0.15, 0.08, 0.40, 0.03), facecolor="dimgray")
    controls = [
        Button(pause_axis, "Pause", color="dimgray", hovercolor="gray"),
        Button(reset_axis, "Reset", color="dimgray", hovercolor="gray"),
        Slider(steps_axis, "Steps/frame", 1, 20, valinit=1, valstep=1, color="white"),
        Slider(dt_axis, "dt multiplier", 0.25, 4.0, valinit=1.0, color="white"),
    ]
    controls[0].on_clicked(toggle_pause)
    controls[1].on_clicked(reset_simulation)
    controls[2].on_changed(
        lambda value: frame_steps.__setitem__(0, int(value))
    )
    controls[3].on_changed(
        lambda value: time_step.__setitem__(0, float(dt) * float(value))
    )
    for slider in controls[2:]:
        slider.label.set_color("white")
        slider.valtext.set_color("white")

    figure.canvas.mpl_connect("scroll_event", zoom_with_mouse)
    animation = FuncAnimation(figure, update, interval=50, cache_frame_data=False)

    def close(_event) -> None:
        animation.event_source.stop()
        if on_close:
            on_close()

    figure.canvas.mpl_connect("close_event", close)
    plt.show()