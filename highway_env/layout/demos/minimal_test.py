"""Absolute minimum pygame window test -- no highway_env, no road network,
just a moving red square. Run with:

    python minimal_test.py

If this doesn't show a window either, the problem is something about this
specific script/invocation context, not the traffic simulation code.
"""
import pygame


def main():
    pygame.init()
    desktop = pygame.display.Info()
    w, h = int(desktop.current_w * 0.5), int(desktop.current_h * 0.5)
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("minimal test -- close to quit")
    print(f"minimal test window open ({w}x{h})")

    clock = pygame.time.Clock()
    x = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        window.fill((30, 30, 30))
        pygame.draw.rect(window, (255, 0, 0), (x, h // 2, 50, 50))
        x = (x + 5) % w
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
