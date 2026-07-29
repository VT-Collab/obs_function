"""
no_assist.py — Control condition: robot provides no assistance.

The human navigates alone with only their own FOV-limited observations.
Used as the lower bound in the three-way comparison.
"""

from __future__ import annotations
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


class NoAssist:
    """Null assistance policy. Robot is silent the entire episode."""

    def __init__(self):
        self.n_assists: int = 0

    def reset(self, state) -> None:
        self.n_assists = 0

    def step(self, state, human_kb: dict):
        return None  # never intervenes
