"""Run this yourself in a normal terminal (not through an agent/sandbox):

    python watch.py

It opens a real, live-updating window with continuous background traffic
running over one combined scene built entirely from build_scene.py's
primitives: add_four_way, add_three_way, turn_corner, round_about, merge.
Traffic keeps spawning indefinitely (cycling through straight/right/left
turns wherever they exist) until you close the window or press Esc; one
vehicle per merge cycle actually changes lanes from the ramp onto the
highway (drawn in orange) instead of just going straight.

Keys: q / ESC / close window -> quit.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pygame
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle
from build_scene import add_four_way, add_three_way, turn_corner, round_about, merge

CONTENT_W, CONTENT_H = 1200, 800

FOUR_WAY_DEST = {
    0: {"straight": "fw_o2_0", "right": "fw_o3_0", "left": "fw_o1_0"},
    1: {"straight": "fw_o3_0", "right": "fw_o0_0", "left": "fw_o2_0"},
    2: {"straight": "fw_o0_0", "right": "fw_o1_0", "left": "fw_o3_0"},
    3: {"straight": "fw_o1_0", "right": "fw_o2_0", "left": "fw_o0_0"},
}
MOVES = ["straight", "right", "left"]
THREE_WAY_OPTIONS = {
    0: ["tw_o1_0", "tw_o3_0"],   # left, right
    1: ["tw_o0_0", "tw_o3_0"],   # right, straight
    3: ["tw_o0_0", "tw_o1_0"],   # left, straight
}
ROUNDABOUT_EXITS = {0: [2, 1, 3], 1: [3, 2, 0], 2: [0, 3, 1], 3: [1, 0, 2]}


def build_scene():
    net = RoadNetwork()
    FW, TW, TC, RA, MG = (0.0, 0.0), (500.0, 0.0), (1000.0, 0.0), (1500.0, 0.0), (2000.0, -100.0)

    add_four_way(net, center=FW, n_vertical=2, n_horizontal=2, access_length=60.0, prefix="fw_")
    add_three_way(net, center=TW, n_stem=2, n_cross=2, access_length=60.0, prefix="tw_", missing_corner=2)
    turn_corner(net, center=TC, heading_deg=0, n_lanes=2, access_length=60.0, prefix="tc_")
    round_about(net, center=RA, radius=20.0, n_lanes=2, access_length=80.0, prefix="ra_")
    obstacle_pos, obstacle_heading = merge(net, center=MG, heading_deg=0, n_lanes=2,
                                            before_length=60.0, taper_length=40.0,
                                            merge_length=40.0, after_length=60.0, prefix="mg_")

    road = Road(network=net, np_random=np.random.RandomState(0), record_history=False)
    road.objects.append(Obstacle(road, obstacle_pos, heading=obstacle_heading))

    tiles = [
        dict(px=(0, 0, 400, 400), center=FW, half=(90.0, 90.0)),
        dict(px=(400, 0, 400, 400), center=TW, half=(90.0, 90.0)),
        dict(px=(800, 0, 400, 400), center=TC, half=(90.0, 90.0)),
        dict(px=(0, 400, 400, 400), center=RA, half=(100.0, 100.0)),
        dict(px=(400, 400, 800, 400), center=(MG[0] + 100.0, MG[1]), half=(120.0, 60.0)),
    ]
    tile_surfaces = []
    for t in tiles:
        px, py, pw, ph = t["px"]
        hw, hh_ = t["half"]
        surf = WorldSurface((pw, ph), 0, pygame.Surface((pw, ph)))
        surf.scaling = pw / (2 * hw)
        surf.origin = np.array([t["center"][0] - hw, t["center"][1] - hh_])
        tile_surfaces.append(surf)

    slots = []
    for idx, corner in enumerate(range(4)):
        slots.append(dict(from_=f"fw_o{corner}_0", to_=f"fw_ir{corner}_0", lon=10.0, speed=9.0,
                           interval=110, phase=idx * 27, cycle_i=0,
                           options=[FOUR_WAY_DEST[corner][m] for m in MOVES]))
    for idx, corner in enumerate([0, 1, 3]):
        slots.append(dict(from_=f"tw_o{corner}_0", to_=f"tw_ir{corner}_0", lon=10.0, speed=9.0,
                           interval=110, phase=idx * 35, cycle_i=0, options=THREE_WAY_OPTIONS[corner]))
    for i in (0, 1):
        slots.append(dict(from_=f"tc_in_{i}", to_=f"tc_mid_{i}", lon=5.0, speed=8.0,
                           interval=90, phase=i * 45, cycle_i=0, options=[f"tc_out_{i}"]))
    for k in range(4):
        slots.append(dict(from_=f"ra_farin{k}_0", to_=f"ra_bendin{k}_0", lon=5.0, speed=7.0,
                           interval=100, phase=k * 25, cycle_i=0,
                           options=[f"ra_farout{e}_0" for e in ROUNDABOUT_EXITS[k]]))
    for i in (0, 1):
        slots.append(dict(from_=f"mg_a_{i}", to_=f"mg_b_{i}", lon=10.0, speed=9.0,
                           interval=80, phase=i * 40, cycle_i=0, options=[f"mg_d_{i}"]))
    merge_ramp_slot = dict(from_="mg_j", to_="mg_k", lon=5.0, speed=9.0,
                            interval=190, phase=0, cycle_i=0, options=["__merge__"])
    all_slots = slots + [merge_ramp_slot]

    spawn_points = {}
    for slot in all_slots:
        spawn_points[id(slot)] = net.get_lane((slot["from_"], slot["to_"], 0)).position(slot["lon"], 0)

    return road, tiles, tile_surfaces, slots, merge_ramp_slot, spawn_points


def main():
    road, tiles, tile_surfaces, slots, merge_ramp_slot, spawn_points = build_scene()
    state = {"step": 0}

    def spawn(from_node, to_node, longitudinal, dest_node, speed=9.0):
        v = IDMVehicle.make_on_lane(road, (from_node, to_node, 0), longitudinal=longitudinal, speed=speed)
        v.plan_route_to(dest_node)
        v.randomize_behavior()
        v._spawn_step = state["step"]
        road.vehicles.append(v)
        return v

    def maybe_spawn(slot, step):
        if (step - slot["phase"]) % slot["interval"] != 0:
            return
        p = spawn_points[id(slot)]
        if any(np.linalg.norm(v.position - p) < 15.0 for v in road.vehicles):
            return
        dest = slot["options"][slot["cycle_i"] % len(slot["options"])]
        slot["cycle_i"] += 1
        if dest == "__merge__":
            v = spawn(slot["from_"], slot["to_"], slot["lon"], "mg_b_ramp", speed=slot["speed"])
            v._ramp_merge, v._merged = True, False
            v.color = (255, 165, 0)
        else:
            spawn(slot["from_"], slot["to_"], slot["lon"], dest, speed=slot["speed"])

    def update_merging_vehicles():
        for v in road.vehicles:
            if getattr(v, "_ramp_merge", False) and not v._merged and v.lane_index[0] == "mg_b_ramp":
                v.target_lane_index = ("mg_b_1", "mg_c_1", 0)
                v.route = []
                v._merged = True

    def prune(step):
        road.vehicles[:] = [
            v for v in road.vehicles
            if not v.crashed and step - getattr(v, "_spawn_step", step) < 900
        ]

    def render_content(content):
        content.fill((100, 100, 100))
        for tile, surf in zip(tiles, tile_surfaces):
            RoadGraphics.display(road, surf)
            RoadGraphics.display_road_objects(road, surf, offscreen=True)
            RoadGraphics.display_traffic(road, surf, simulation_frequency=20, offscreen=True)
            content.blit(surf, tile["px"][:2])

    pygame.init()
    desktop = pygame.display.Info()
    w, hh = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, hh))
    pygame.display.set_caption("build_scene.py -- live traffic demo")
    content = pygame.Surface((CONTENT_W, CONTENT_H))
    offset = ((w - CONTENT_W) // 2, (hh - CONTENT_H) // 2)

    dt = 1 / 20
    clock = pygame.time.Clock()
    running = True
    print(f"live window open ({w}x{hh}) -- close it (or press q/Esc) to stop")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        for slot in slots:
            maybe_spawn(slot, state["step"])
        maybe_spawn(merge_ramp_slot, state["step"])
        update_merging_vehicles()
        road.act()
        road.step(dt)
        prune(state["step"])
        state["step"] += 1

        render_content(content)
        window.fill((30, 30, 30))
        window.blit(content, offset)
        pygame.display.flip()
        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()
