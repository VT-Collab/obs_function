"""
MISHA NEW CHANGE - subtask -> station-kind lookup for the from-scratch
limited-vision steak human (fov/human/agent/limited_vision_human.py).

The AGENT owns belief, decision AND execution: it routes with its OWN BFS over
cells it has actually SEEN (_bfs_step), and it acts on its OWN discovered stations.
All this planner still provides is the static subtask -> station-KIND map, so the
human can look up which kind a subtask targets and then consult its own
`self.stations`. NO map or tile data lives here - nothing about the planner can
leak an unseen tile into the limited-vision human.

(The full-map motion planning that used to live here - ground-truth station
positions from the mdp + mlp.mp.get_plan over the whole map - was the largest
cheat in the agent and has been removed. The ROBOT, which legitimately has full
world vision, calls mlp.mp.get_plan directly in its own env wrapper.)
"""

# Where each subtask must be performed. 'check_*' subtasks are the distinctive
# ones: they exist only because a belief went stale, so they appear in a narrow
# FOV's trajectory and not a wide one's.
SUBTASK_TARGETS = {
    "check_pot": "pot",
    "check_board": "board",
    "check_sink": "sink",
    "pickup_meat": "meat",
    "pickup_onion": "onion",
    "pickup_plate": "dish",
    "drop_meat": "pot",
    "drop_onion": "board",
    "drop_plate": "sink",
    "chop_onion": "board",
    "heat_hot_plate": "sink",
    "pickup_steak": "pot",
    "pickup_garnish": "board",
    "pickup_hot_plate": "sink",
    "deliver": "serve",
    # Somewhere legal to set an item down when its station is occupied.
    "dump_item": "counter",
}


class SteakMotionPlanner:
    """Constructed `SteakMotionPlanner(mdp, mlp)` throughout, but the human uses
    ONLY target_kind. mdp/mlp are held for callers that pass the planner on to
    full-vision routing (robot / inference shadows); they are never used to route
    the limited-vision human, which pathfinds over seen floor by itself."""

    def __init__(self, mdp, mlp):
        self.mdp = mdp
        self.mlp = mlp

    def target_kind(self, subtask):
        """Which station kind this subtask acts on. Pure lookup - no map data."""
        return SUBTASK_TARGETS.get(subtask, "")
