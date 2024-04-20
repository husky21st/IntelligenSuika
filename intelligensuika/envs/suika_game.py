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
ONE_SECOND_FRAME = 60
FRUIT_COLOR_SIZE = [
    ((220, 20, 60), 20),     # Crimson
    ((250, 128, 114), 22),   # Salmon
    ((186, 85, 211), 24),    # Medium Orchid
    ((255, 165, 0), 26),     # Orange
    ((255, 140, 0), 28),     # Dark Orange
    ((255, 0, 0), 30),       # Red
    ((240, 230, 140), 32),   # Khaki
    ((255, 192, 203), 34),   # Pink
    ((255, 255, 0), 36),     # Yellow
    ((173, 255, 47), 38),    # Green Yellow
    ((0, 128, 0), 40)        # Green
]

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
    def __init__(self, center, selected_fruit):
        self.pos = Vec2(center)
        self.color  = selected_fruit[0]
        self.radius = selected_fruit[1]
        self.v = Vec2(0, 0)
        self.m = 1

    def update(self, delta):
        self.v.y += GRAVITY*ONE_SECOND_FRAME* delta  # Apply gravity
        self.pos += self.v * delta  # Update position based on velocity

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

# Create walls and circles list
walls = [Line((200, 550), (650, 550)), Line((200, 550), (200, 100)), Line((650, 550), (650, 100))]
circles = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            circles.append(PhysicsCircle(event.pos, random.choice(FRUIT_COLOR_SIZE)))

    delta_time = clock.get_time() / 1000

    for i, circle in enumerate(circles):
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
        
        # Circle collision
        for other_circle in circles[i+1:]:
            dist = (circle.pos - other_circle.pos).length()
            if dist < circle.radius + other_circle.radius:
                overlap = circle.radius + other_circle.radius - dist
                normal = normalized(circle.pos - other_circle.pos)
                move_dist = overlap / 2
                circle.pos += normal * move_dist
                other_circle.pos -= normal * move_dist
                # Adjust velocities based on mass
                v1 = circle.v
                v2 = other_circle.v
                m1 = circle.m
                m2 = other_circle.m
                circle.v = v1 - 2 * m2 / (m1 + m2) * dot(v1 - v2, normal) * normal * RESTITUTION
                other_circle.v = v2 - 2 * m1 / (m1 + m2) * dot(v2 - v1, -normal) * -normal * RESTITUTION

    # Draw everything
    screen.fill((0, 0, 0))
    for circle in circles:
        circle.draw()
    for wall in walls:
        wall.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
