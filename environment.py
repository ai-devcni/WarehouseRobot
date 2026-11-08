"""A very small warehouse-floor environment for a robot that has to find a dock.

The layout is kept intentionally simple: a few fixed barriers, a single goal,
and a robot that only sees a handful of range readings ahead of it.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import pygame


class RobotEnv:
    WIDTH, HEIGHT = 800, 600
    ROBOT_RADIUS = 12
    GOAL_RADIUS = 14
    MAX_RAY_DISTANCE = 180.0
    RAY_ANGLES = (-45.0, -22.5, 0.0, 22.5, 45.0)
    TURN_ANGLE = math.radians(12)
    MOVE_DISTANCE = 6.0

    def __init__(self, render: bool = True, seed: Optional[int] = None):
        self.render_enabled = render
        self.rng = random.Random(seed)
        self.screen = None
        self.clock = None

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Warehouse Robot")
            self.clock = pygame.time.Clock()

        self.obstacles = [
            pygame.Rect(150, 110, 150, 28),
            pygame.Rect(420, 120, 30, 180),
            pygame.Rect(170, 285, 190, 30),
            pygame.Rect(535, 370, 150, 28),
        ]
        self.episode_steps = 0
        self.reset()

    def reset(self):
        self.x, self.y = 70.0, 70.0
        self.heading = 0.0
        self.episode_steps = 0

        while True:
            gx = self.rng.randint(60, self.WIDTH - 60)
            gy = self.rng.randint(60, self.HEIGHT - 60)
            goal_rect = pygame.Rect(
                gx - self.GOAL_RADIUS,
                gy - self.GOAL_RADIUS,
                self.GOAL_RADIUS * 2,
                self.GOAL_RADIUS * 2,
            )
            if math.hypot(gx - self.x, gy - self.y) > 250 and not any(
                rect.colliderect(goal_rect) for rect in self.obstacles
            ):
                self.goal = (float(gx), float(gy))
                break

        return self.get_state()

    def _solid(self, x: float, y: float) -> bool:
        r = self.ROBOT_RADIUS
        if x - r < 0 or x + r >= self.WIDTH or y - r < 0 or y + r >= self.HEIGHT:
            return True
        return any(rect.inflate(2 * r, 2 * r).collidepoint(x, y) for rect in self.obstacles)

    def _ray_distance(self, angle: float) -> float:
        """Cast a sensor ray in front of the robot until it hits a wall or obstacle."""
        dx, dy = math.cos(angle), math.sin(angle)
        distance = 0.0

        while distance < self.MAX_RAY_DISTANCE:
            distance += 2.0
            if self._solid(self.x + dx * distance, self.y + dy * distance):
                return min(distance, self.MAX_RAY_DISTANCE)

        return self.MAX_RAY_DISTANCE

    def ray_distances(self):
        return [
            self._ray_distance(self.heading + math.radians(angle)) / self.MAX_RAY_DISTANCE
            for angle in self.RAY_ANGLES
        ]

    def get_state(self):
        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        relative = (bearing - self.heading + math.pi) % (2 * math.pi) - math.pi

        return self.ray_distances() + [
            distance / math.hypot(self.WIDTH, self.HEIGHT),
            relative / math.pi,
        ]

    def step(self, action: int):
        old_distance = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)

        if action == 0:
            self.heading -= self.TURN_ANGLE
        elif action == 1:
            nx = self.x + math.cos(self.heading) * self.MOVE_DISTANCE
            ny = self.y + math.sin(self.heading) * self.MOVE_DISTANCE
            self.x, self.y = nx, ny
        elif action == 2:
            self.heading += self.TURN_ANGLE
        else:
            raise ValueError("action must be 0 (left), 1 (forward), or 2 (right)")

        self.episode_steps += 1
        new_distance = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        done = False

        reward = -1.0 + (old_distance - new_distance) / self.MOVE_DISTANCE
        if self._solid(self.x, self.y):
            reward, done = -100.0, True
        elif new_distance <= self.GOAL_RADIUS + self.ROBOT_RADIUS:
            reward, done = 100.0, True

        return self.get_state(), reward, done, {"distance": new_distance}

    def render(self):
        if not self.render_enabled:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((30, 34, 42))
        pygame.draw.rect(self.screen, (220, 224, 230), (0, 0, self.WIDTH, self.HEIGHT), 3)

        for rect in self.obstacles:
            pygame.draw.rect(self.screen, (105, 112, 126), rect, border_radius=4)

        pygame.draw.circle(self.screen, (70, 220, 120), tuple(map(int, self.goal)), self.GOAL_RADIUS)

        for offset in self.RAY_ANGLES:
            angle = self.heading + math.radians(offset)
            distance = self._ray_distance(angle)
            end = (int(self.x + math.cos(angle) * distance), int(self.y + math.sin(angle) * distance))
            colour = (70, 230, 100) if distance >= self.MAX_RAY_DISTANCE - 1 else (240, 80, 75)
            pygame.draw.line(self.screen, colour, (int(self.x), int(self.y)), end, 1)

        pygame.draw.circle(self.screen, (75, 155, 255), (int(self.x), int(self.y)), self.ROBOT_RADIUS)
        nose = (int(self.x + math.cos(self.heading) * 20), int(self.y + math.sin(self.heading) * 20))
        pygame.draw.line(self.screen, (255, 255, 255), (int(self.x), int(self.y)), nose, 3)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_enabled:
            pygame.quit()
