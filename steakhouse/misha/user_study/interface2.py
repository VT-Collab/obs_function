"""Debug variant of interface.py -- SAME human-click gameplay, SAME
belief-gated menu, SAME FOV-limited HumanBehavior. The only thing that
changes is what gets DRAWN.

    python -m user_study.interface2 --layout back_bar --partner fov

interface.py's game-view panel is the human's own fog-of-war
(View.human_seat()): the robot is genuinely invisible whenever it is
outside your cone, which is exactly right for the real study but makes it
hard to see what a teammate policy is even doing while testing or tuning
one. This file changes nothing about gameplay or inference -- HumanBehavior,
legal_menu(), SubtaskFOVPosterior and FOVFilter all read ep.human/
ep.debug_post directly and never touch ep.view, which App._panel()'s own
docstring notes is a pure rendering cache. It overrides App._panel() to
draw View.robot_seat() instead: full ground truth, always, with a
translucent tint (not a hide) over whatever is outside the human's cone --
the same panel watch.py shows by default. Your own menu stays exactly as
fov-gated as it is in interface.py; only the picture changes.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # .../user_study
MISHA = os.path.dirname(HERE)                                  # .../misha
sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, MISHA)

from user_study.interface import App, parse_args               # noqa: E402


class DebugApp(App):
    """App, but the game-view panel is always the omniscient one -- see
    this module's own docstring for exactly what stays FOV-gated (your
    menu) versus what does not any more (the picture)."""

    def _panel(self, ep):
        return ep.view.robot_seat()

    def _view_caption(self, ep):
        return ("fov %d -- DEBUG VIEW: robot always shown, grey = outside "
                "your cone (your own choices are still fov-gated)"
                % ep.cfg["fov"])


if __name__ == "__main__":
    DebugApp(parse_args()).run()
