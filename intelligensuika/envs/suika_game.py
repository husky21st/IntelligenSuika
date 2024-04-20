import pygame
import math
import random
from pygame.math import Vector2 as Vec2

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True

# Constants
GRAVITY = 9.8
RESTITUTION = 0.8  # Restitution coefficient for collisions

# Helper Functions
def add(v1, v2):
    return Vec2(v1[0] + v2[0], v1[1] + v2[1])

def sub(v1, v2):
    return Vec2(v1[0] - v2[0], v1[1] - v2[1])

def mul(v, scalar):
    return Vec2(v[0] * scalar, v[1] * scalar)

def div(v, scalar):
    return Vec2(v[0] / scalar, v[1] / scalar)

def length(v):
    return v.length()

def normalized(v):
    if v.length() == 0:
        return Vec2(0, 0)
    return v.normalize()

def dot(v1, v2):
    return v1.dot(v2)

class Line:
    def __init__(self, start, end):
        self.start = Vec2(start)
        self.end = Vec2(end)

    def draw(self):
        pygame.draw.line(screen, (128, 128, 128), self.start, self.end)

    def closest(self, point):
        line_vec = self.end - self.start
        point_vec = point - self.start
        t = max(0, min(1, point_vec.dot(line_vec) / line_vec.length_squared()))
        return self.start + line_vec * t

class PhysicsCircle:
    def __init__(self, center, radius):
        self.pos = Vec2(center)
        self.radius = radius
        self.v = Vec2(0, 0)
        self.m = 1

    def update(self, delta):
        self.v.y += GRAVITY* 60 * delta  # Apply gravity
        self.pos += self.v * delta  # Update position based on velocity

    def draw(self):
        pygame.draw.circle(screen, (255, 255, 255), (int(self.pos.x), int(self.pos.y)), self.radius)

# Create walls and circles list
walls = [Line((100, 500), (700, 500)), Line((100, 700), (100, 100)), Line((700, 700), (700, 100))]
circles = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            circles.append(PhysicsCircle(event.pos, random.randint(10, 100)))

    delta_time = clock.get_time() / 1000

    for circle in circles:
        circle.update(delta_time)
        # Wall collision
        for wall in walls:
            closest_point = wall.closest(circle.pos)
            if (closest_point - circle.pos).length() < circle.radius:
                overlap = circle.radius - (closest_point - circle.pos).length()
                normal = normalized(circle.pos - closest_point)
                circle.pos += normal * overlap
                # Reflect velocity
                circle.v -= 2 * circle.v.dot(normal) * normal * RESTITUTION

    # Draw everything
    screen.fill((0, 0, 0))
    for circle in circles:
        circle.draw()
    for wall in walls:
        wall.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
