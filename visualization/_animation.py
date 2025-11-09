"""Shared helpers for building matplotlib animations."""

from collections.abc import Callable, Sequence
from typing import Optional

from matplotlib.animation import FuncAnimation

PanelUpdater = Callable[[int], None]
MessageFactory = Callable[[int], str]


def run_animation(
    fig,
    simulator,
    panels: Sequence[PanelUpdater],
    frames: int,
    interval: int,
    message_fn: Optional[MessageFactory] = None,
):
    """Drive a simulation forward while updating associated panels."""

    def animate(frame: int):
        if message_fn is not None:
            print(message_fn(frame))

        simulator.step()

        for update in panels:
            update(frame)

        return []

    return FuncAnimation(fig, animate, frames=frames, interval=interval)
