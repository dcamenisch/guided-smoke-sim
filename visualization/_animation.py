"""Shared helpers for building matplotlib animations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from simulation.simulator import SmokeSimulator

PanelUpdater = Callable[[int], None]
MessageFactory = Callable[[int], str]


def run_animation(
    fig: Figure,
    simulator: SmokeSimulator,
    panels: Sequence[PanelUpdater],
    frames: int,
    interval: int,
    message_fn: Optional[MessageFactory] = None,
) -> FuncAnimation:
    """Drive a simulation forward while updating associated panels."""

    def animate(frame: int) -> list[object]:
        if message_fn is not None:
            print(message_fn(frame))

        simulator.step()

        for update in panels:
            update(frame)

        return []

    return FuncAnimation(fig, animate, frames=frames, interval=interval)
