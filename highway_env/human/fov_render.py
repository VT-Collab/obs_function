"""Shared rendering helpers for watch.py and play.py: a real 2D shadow-cast
visibility mask (not a wireframe) and a rotating chase camera. No argparse,
no main loop, no simulation stepping -- pure rendering, imported by both.

--------------------------------------------------------------------------
THE MASK -- a real 2D shadow-cast visibility overlay
--------------------------------------------------------------------------
A semi-transparent (or fully opaque, per caller) tinted layer covers the
whole frame; the FOV cone (or a full circle if the cone is disabled) is cut
out of it as fully transparent, then a shadow polygon is drawn back in
(re-tinted) behind every occluding vehicle within that cone. The shadow of
a convex rectangle as seen from a point is bounded by the two corners that
are angularly extreme relative to that point (`_shadow_polygon` below) --
a standard 2D shadow-casting construction. Occlusion is therefore visible
directly (things go dark), matching how the actual perception logic in
limit_vision_human.py decides visibility.

Two callers, two different tints -- mirroring steakhouse/misha's watch.py
(the human's blind area GREYED, moderate alpha, "you see it all, grey is
what the human can't") vs. play.py's human seat (the seat you're actually
IN, fully opaque BLACK, "everything outside your cone is BLACK"). Which
tint/alpha to use is the caller's choice (`draw_fov_mask`'s `color`/
`mask_alpha` args); this module doesn't hardcode either.

--------------------------------------------------------------------------
THE CHASE CAMERA -- rotate the render, not the world
--------------------------------------------------------------------------
highway-env's WorldSurface only supports translation + uniform scaling
(pos2pix is a pure offset-and-scale, no rotation term) -- there's no
built-in way to render the map itself already rotated. So instead: render
the world into a SQUARE buffer, centered on the human, big enough that no
corner of the final window is left uncovered after an arbitrary rotation
(`chase_camera_buffer`'s `reach` computation -- the classic "render bigger
than you need, then rotate and crop" trick); rotate that whole buffer with
pygame.transform.rotate(); then blit it so the human's fixed position in
the buffer (its center, since pygame rotates around a surface's center)
lands at a fixed anchor point on screen.

The rotation ANGLE, derived and verified empirically (checked with a known
marker point, not assumed): highway-env's pixel mapping does not flip the y
axis (pos2pix: pix(x-ox), pix(y-oy), no sign flip), so world heading H,
rendered directly, appears CLOCKWISE-increasing on screen -- heading 0
points screen-right, heading +90deg points screen-down. To make the CURRENT
heading appear "up" (screen -y) after rotating, the buffer must be rotated
by `degrees(H) + 90` (pygame.transform.rotate's positive angle is counter-
clockwise as the viewer sees it). Verified: a marker placed at heading-0's
direction from center, rotated by this formula for H=0, lands directly
above center in the output pixels.
"""
import numpy as np
import pygame

CONE_RADIUS = 35.0  # matches apply_human_aware_car_following's own perception radius
CONE_SEGMENTS = 28
VISIBLE_COLOR = (80, 220, 80)
HIDDEN_COLOR = (220, 70, 70)


# -- shadow-cast visibility mask ---------------------------------------------

def _rect_corners(position, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    dx, dy = length / 2.0, width / 2.0
    local = ((dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy))
    return [position + np.array([c * lx - s * ly, s * lx + c * ly]) for lx, ly in local]


def _extend_point(origin, point, radius):
    delta = point - origin
    dist = np.linalg.norm(delta)
    if dist < 1e-6 or dist >= radius:
        return point
    return origin + delta * (radius / dist)


def _shadow_polygon(origin, corners, radius, pad=0.0):
    """The two corners of a convex quadrilateral that are angularly extreme
    as seen from `origin` bound its silhouette; extending rays through them
    out to `radius` gives the region it occludes beyond itself.

    `pad` (meters, default 0 -- off) would widen that silhouette outward
    before casting the shadow, to fully cover a target vehicle whose own
    body pokes slightly past the blocker's exact silhouette right at the
    boundary (is_occluded() only checks the target's CENTER point, not its
    whole body). Tried at 1.5m and reverted: the shadow is the visible
    proxy for "how big is this blocker", and padding it made an occluder
    look visibly wider/thicker than the vehicle actually rendered there --
    a real, everyday-visible mismatch, versus the boundary-sliver case it
    was fixing, which only shows up for another vehicle sitting almost
    exactly at the edge of being hidden. Matching the blocker's true
    silhouette was judged the better tradeoff; pass pad>0 to restore the
    old behavior if the sliver case turns out to matter more later."""
    rel = np.array([np.arctan2(c[1] - origin[1], c[0] - origin[0]) for c in corners])
    unwrapped = ((rel - rel[0] + np.pi) % (2 * np.pi)) - np.pi
    i_min, i_max = int(np.argmin(unwrapped)), int(np.argmax(unwrapped))
    c_min, c_max = corners[i_min], corners[i_max]

    def _push_outward(point, sign):
        delta = point - origin
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return point
        tangent = np.array([-delta[1], delta[0]]) / dist  # perpendicular, angle-increasing direction
        return point + sign * pad * tangent

    c_min = _push_outward(c_min, -1.0)  # push toward smaller angle -> widens the "min" side outward
    c_max = _push_outward(c_max, 1.0)   # push toward larger angle -> widens the "max" side outward
    far_min = _extend_point(origin, c_min, radius)
    far_max = _extend_point(origin, c_max, radius)
    return [c_min, far_min, far_max, c_max]


def _cone_polygon(position, heading, fov_deg, radius, segments=CONE_SEGMENTS):
    if fov_deg >= 360:
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        return [position + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
    half = np.radians(fov_deg / 2.0)
    angles = np.linspace(heading - half, heading + half, segments)
    return [position] + [position + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]


def draw_fov_mask(surface, human, candidates, fov_deg, enable_fov, enable_occlusion,
                   mask_alpha, color=(0, 0, 0), radius=CONE_RADIUS, pad=0.0, shadow_reach_factor=10.0):
    """Tint everything the human can't currently see, in world space (so it
    rotates correctly along with the world in chase mode -- call this on
    the SAME buffer that later gets rotated, before the rotation happens).

    `color`/`mask_alpha` are the caller's choice: watch.py passes a grey
    tint at moderate alpha (matches steakhouse/misha's BLIND_GREY -- you
    see everything, dimmed where the human can't), play.py passes black at
    full alpha (matches steakhouse/misha's human seat -- genuinely hidden).

    `pad`: see _shadow_polygon's own docstring -- 0 by default, so each
    shadow matches the occluding vehicle's own true size instead of
    rendering it as visibly wider/thicker than it actually is.

    `shadow_reach_factor`: each shadow polygon's own far edge is a
    STRAIGHT chord between two points on the radius-`radius` circle (its
    two, now-padded, angularly-extreme corners extended outward) -- any
    chord between two points on a circle sits strictly INSIDE that circle,
    so that flat edge falls a little short of the light cone's own true
    (many-segment-sampled, effectively curved) outer boundary. The gap
    between them is a small but real unshadowed sliver of light right at
    the rim of the cone, exactly where the shadow should still be covering
    it. Casting the shadow out to `radius * shadow_reach_factor` instead
    of just `radius` pushes that same, unavoidably-flat chord far past
    where the cone itself is even drawn -- nothing rendered depends on the
    shadow polygon's OWN edge being circular, only on it reaching at least
    as far as the light cone everywhere, so a generous multiple removes
    the visible mismatch instead of trying to shape the chord to match an
    arc it geometrically can't.
    """
    if not enable_fov and not enable_occlusion:
        return  # true ablation: no restriction, nothing to tint

    r, g, b = color
    mask = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    mask.fill((r, g, b, mask_alpha))

    light_world = _cone_polygon(human.position, human.heading,
                                 fov_deg if enable_fov else 360.0, radius)
    pygame.draw.polygon(mask, (0, 0, 0, 0), [surface.vec2pix(p) for p in light_world])

    if enable_occlusion:
        for other in candidates:
            if other is human:
                continue
            length = getattr(other, "LENGTH", 5.0)
            width = getattr(other, "WIDTH", 2.0)
            corners = _rect_corners(other.position, other.heading, length, width)
            shadow_world = _shadow_polygon(human.position, corners, radius * shadow_reach_factor, pad=pad)
            pygame.draw.polygon(mask, (r, g, b, mask_alpha), [surface.vec2pix(p) for p in shadow_world])

    surface.blit(mask, (0, 0))


def redraw_ego(surface, human, offscreen=True):
    """Redraw the human's OWN sprite on top of the mask.

    The mask's cone starts exactly at human.position, but the car's body
    extends in every direction from that point -- its rear half falls
    outside the wedge and was getting blacked out mid-vehicle, a visibly
    "cut in half" artifact. You always know what your own car looks like
    regardless of FOV, so it's simplest and most robust to just draw it
    again, on top of the mask, rather than trying to carve an exact
    vehicle-shaped hole out of the mask polygon for this one special case.
    """
    from highway_env.vehicle.graphics import VehicleGraphics  # local import: keeps this module import-light

    VehicleGraphics.display(human, surface, offscreen=offscreen)


def redraw_partial(surface, human, candidates, fov_deg, enable_fov, enable_occlusion,
                    radius=CONE_RADIUS, shadow_reach_factor=10.0, offscreen=True):
    """Redraw each OTHER vehicle's sprite clipped to exactly the part of
    its own body that's genuinely visible -- fully, partially (straddling
    the cone's edge, or poking out from behind another vehicle), or not at
    all -- instead of the previous redraw_visible's all-or-nothing whole-
    sprite toggle keyed off human.visible_candidates() (a single center-
    point test that can only say "this whole vehicle is in or out").

    For each `other`: builds a small per-vehicle "keep" alpha mask (cone
    cutout, then re-hidden behind every OTHER candidate's shadow --
    deliberately never `other`'s own, which is what fixes the self-shadow
    issue redraw_visible also fixed, but here as a natural side effect of
    excluding self rather than an all-or-nothing override), renders
    `other`'s sprite into a same-sized buffer, multiplies the sprite's
    alpha by the keep mask's alpha (pygame.BLEND_RGBA_MULT), then blits
    the result -- so only the genuinely-visible pixels of `other` survive.

    Confined to a small buffer sized to `other`'s own bounding diagonal
    (not the whole frame) for performance: this runs once per nearby
    vehicle per frame.
    """
    from highway_env.road.graphics import WorldSurface  # local import: keeps this module import-light
    from highway_env.vehicle.graphics import VehicleGraphics

    for other in candidates:
        if other is human:
            continue
        length = getattr(other, "LENGTH", 5.0)
        width = getattr(other, "WIDTH", 2.0)
        diag = float(np.hypot(length, width))
        half_px = int(surface.pix(diag)) + 6
        box = half_px * 2
        if box <= 0:
            continue

        cx, cy = surface.vec2pix(other.position)
        buf = WorldSurface((box, box), pygame.SRCALPHA, pygame.Surface((box, box), pygame.SRCALPHA))
        buf.fill((0, 0, 0, 0))
        buf.scaling = surface.scaling
        buf.origin = surface.origin + (np.array([cx, cy]) - half_px) / surface.scaling

        keep = pygame.Surface((box, box), pygame.SRCALPHA)
        if enable_fov:
            keep.fill((255, 255, 255, 0))
            light_world = _cone_polygon(human.position, human.heading, fov_deg, radius)
            pygame.draw.polygon(keep, (255, 255, 255, 255), [buf.vec2pix(p) for p in light_world])
        else:
            keep.fill((255, 255, 255, 255))

        if enable_occlusion:
            for blocker in candidates:
                if blocker is other or blocker is human:
                    continue
                b_length = getattr(blocker, "LENGTH", 5.0)
                b_width = getattr(blocker, "WIDTH", 2.0)
                corners = _rect_corners(blocker.position, blocker.heading, b_length, b_width)
                shadow_world = _shadow_polygon(human.position, corners, radius * shadow_reach_factor)
                pygame.draw.polygon(keep, (0, 0, 0, 0), [buf.vec2pix(p) for p in shadow_world])

        sprite_buf = pygame.Surface((box, box), pygame.SRCALPHA)
        VehicleGraphics.display(other, buf, offscreen=True)
        sprite_buf.blit(buf, (0, 0))
        sprite_buf.blit(keep, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(sprite_buf, (cx - half_px, cy - half_px))


def draw_debug_lines(surface, human, visible_ids, candidates):
    for other in candidates:
        color = VISIBLE_COLOR if id(other) in visible_ids else HIDDEN_COLOR
        pygame.draw.line(surface, color, surface.vec2pix(human.position), surface.vec2pix(other.position), 1)


# -- chase camera -------------------------------------------------------

def chase_camera_buffer(human, scaling, window_size, anchor):
    """A square WorldSurface, centered on the human, sized so that after an
    arbitrary rotation and a crop/blit pinning the human at `anchor` in a
    `window_size` window, every corner of that window is still covered."""
    from highway_env.road.graphics import WorldSurface  # local import: keeps this module import-light

    w, hh = window_size
    ax, ay = anchor
    reach = max(np.hypot(ax, ay), np.hypot(w - ax, ay),
                np.hypot(ax, hh - ay), np.hypot(w - ax, hh - ay))
    buf_px = int(2 * reach * 1.05) + 2
    buf = WorldSurface((buf_px, buf_px), 0, pygame.Surface((buf_px, buf_px)))
    buf.scaling = scaling
    buf.origin = human.position - np.array([buf_px / 2.0, buf_px / 2.0]) / scaling
    return buf


def render_chase_frame(buf, human, window_size, anchor, bg_color):
    """Rotate `buf` so human.heading points up (see module docstring for
    the angle derivation) and blit it centered at `anchor` into a new
    window_size surface."""
    rotate_deg = np.degrees(human.heading) + 90.0
    rotated = pygame.transform.rotate(buf, rotate_deg)
    rw, rh = rotated.get_size()
    frame = pygame.Surface(window_size)
    frame.fill(bg_color)
    frame.blit(rotated, (anchor[0] - rw / 2.0, anchor[1] - rh / 2.0))
    return frame


def chase_screen_pos(human, world_pos, scale, anchor):
    """Where `world_pos` ends up on the FINAL chase-camera frame (the
    thing render_chase_frame returns), without tracing back through
    chase_camera_buffer's own buf.vec2pix + pygame.transform.rotate's own
    pixel math (whose rotation-direction convention isn't worth depending
    on here). Derived straight from what the chase camera is DEFINED to
    do: the human always sits exactly at `anchor`, and human.heading
    always points to screen-up. Solve for the one rotation phi that sends
    the human's own world heading vector to screen-up (0, -1) in pygame's
    x-right/y-down convention -- phi = -pi/2 - human.heading -- and apply
    that same rotation to any other point's own offset from the human.
    Verified algebraically: world_pos = human.position maps to exactly
    `anchor` (rel=0), and a point straight ahead of the human maps to
    directly above anchor on screen, both required by the camera's own
    definition."""
    phi = -np.pi / 2.0 - human.heading
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    rel = np.asarray(world_pos, dtype=float) - np.asarray(human.position, dtype=float)
    rx = cos_p * rel[0] - sin_p * rel[1]
    ry = sin_p * rel[0] + cos_p * rel[1]
    return anchor[0] + scale * rx, anchor[1] + scale * ry
