"""Renders every real_NNN_rebuilt (001-010) and its real_NNN_regulated_road
sibling, and reports whether swapping Road for RegulatedRoad changes a
single pixel, for all ten layouts at once. See real_001_regulated_road.py's
own docstring for why this SHOULD be a no-op (argued from what
RoadGraphics.display/draw_routes actually read) -- this is the
measurement, not the argument.

Two separate comparisons per layout, kept distinct on purpose:
  1. A FRESH real_NNN_rebuilt render (via this same render_offscreen call,
     right now, at that layout's own reference PNG's own pixel size --
     these are NOT all the same size, real_001 is 2800x2800 but 002-010
     are 1600x1200, confirmed directly) against the on-disk reference
     PNG -- sanity-checks that this script's own render parameters
     actually reproduce that reference, so comparison 2 means something.
     If this one doesn't match, the reference was generated some other
     way (different params/pygame version) and comparison 2 is the only
     one that actually answers the Road-vs-RegulatedRoad question.
  2. The FRESH real_NNN_rebuilt render against a real_NNN_regulated_road
     render, both made identically by this same script in the same
     process -- this is the actual Road-vs-RegulatedRoad test, immune to
     any "how was the old reference generated" uncertainty in (1).

Writes every fresh render next to (not over) the existing references so
they can be opened and looked at, never touches anything in
layout/rebuilt_renders/ itself.

    python _compare_regulated_render.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # layouts/ itself
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # layout/
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import display_all as d  # noqa: E402

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rebuilt_renders")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LAYOUT_NUMBERS = [f"{n:03d}" for n in range(1, 11)]


def render_to_png(name, size, out_path):
    surface = d.render_offscreen(name, size=size)
    if surface is None:
        raise RuntimeError(f"{name} failed to build/render -- see printed error above")
    pygame.image.save(surface, out_path)
    return out_path


def compare_files(path_a, path_b, label):
    """Returns True iff the two files are pixel-identical (regardless of
    whether the raw PNG bytes matched exactly -- encoding/metadata
    differences that don't touch actual image content still count as a
    pass here)."""
    bytes_a = open(path_a, "rb").read()
    bytes_b = open(path_b, "rb").read()
    if bytes_a == bytes_b:
        print(f"  {label}: BYTE-FOR-BYTE IDENTICAL ({len(bytes_a)} bytes)")
        return True
    surf_a = pygame.image.load(path_a)
    surf_b = pygame.image.load(path_b)
    if surf_a.get_size() != surf_b.get_size():
        print(f"  {label}: DIFFERENT SIZE {surf_a.get_size()} vs {surf_b.get_size()} -- cannot compare pixels")
        return False
    px_a = pygame.image.tostring(surf_a, "RGB")
    px_b = pygame.image.tostring(surf_b, "RGB")
    if px_a == px_b:
        print(f"  {label}: PIXELS IDENTICAL (bytes differ by {len(bytes_a)}b vs {len(bytes_b)}b -- "
              f"PNG encoding/metadata only, not image content)")
        return True
    n_diff = sum(1 for x, y in zip(px_a, px_b) if x != y)
    print(f"  {label}: PIXELS DIFFER -- {n_diff} of {len(px_a)} raw channel bytes differ")
    return False


def main():
    pygame.init()
    results = []

    for num in LAYOUT_NUMBERS:
        rebuilt_name = f"real_{num}_rebuilt"
        regulated_name = f"real_{num}_regulated_road"
        reference_png = os.path.join(REFERENCE_DIR, f"{rebuilt_name}.png")
        print(f"=== real_{num} ===")

        ref_surf = pygame.image.load(reference_png)
        size = ref_surf.get_size()

        fresh_rebuilt = os.path.join(OUT_DIR, f"_fresh_{rebuilt_name}.png")
        fresh_regulated = os.path.join(OUT_DIR, f"_fresh_{regulated_name}.png")
        render_to_png(rebuilt_name, size, fresh_rebuilt)
        render_to_png(regulated_name, size, fresh_regulated)

        ok1 = compare_files(reference_png, fresh_rebuilt,
                             "(1) on-disk reference vs. fresh rebuilt render")
        ok2 = compare_files(fresh_rebuilt, fresh_regulated,
                             "(2) fresh rebuilt vs. fresh regulated_road [the actual test]")
        results.append((num, ok1, ok2))
        print()

    print("=== SUMMARY ===")
    print(f"{'layout':<12}{'(1) matches old reference':<28}{'(2) Road == RegulatedRoad render':<35}")
    for num, ok1, ok2 in results:
        print(f"real_{num:<7}{'yes' if ok1 else 'NO':<28}{'yes' if ok2 else 'NO':<35}")
    n_ok2 = sum(1 for _, _, ok2 in results if ok2)
    print(f"\n(2) passed for {n_ok2}/{len(results)} layouts -- this is the one that answers "
          f"\"does RegulatedRoad change the render\"")


if __name__ == "__main__":
    main()
