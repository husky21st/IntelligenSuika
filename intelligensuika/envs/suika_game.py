import pygame
import math
import random
from pygame.math import Vector2 as Vec2
from setting import *
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True

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
    def __init__(self, center, fruit_index):
        self.pos = Vec2(center)
        self.fruit_index = fruit_index
        self.color  = FRUIT_COLOR_SIZE[fruit_index][0]
        self.radius = FRUIT_COLOR_SIZE[fruit_index][1]
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

def handle_collisions(circles):
    i = 0
    while i < len(circles):
        j = i + 1
        while j < len(circles):
            circle1, circle2 = circles[i], circles[j]
            if (circle1.pos - circle2.pos).length() < (circle1.radius + circle2.radius):
                if circle1.fruit_index == circle2.fruit_index and circle1.fruit_index < len(FRUIT_COLOR_SIZE) - 1:
                    new_index = circle1.fruit_index + 1
                    new_pos = (circle1.pos + circle2.pos) * 0.5
                    new_circle = PhysicsCircle(new_pos, new_index)
                    new_circle.v = (circle1.v * circle1.m + circle2.v * circle2.m) / (circle1.m + circle2.m)
                    circles.append(new_circle)
                    circles.pop(j)  # Remove second circle first
                    circles.pop(i)  # Remove first circle
                    i -= 1  # Adjust index after removal
                    break
                elif circle1.fruit_index == circle2.fruit_index and circle1.fruit_index == len(FRUIT_COLOR_SIZE) - 1:
                    circles.pop(j)
                    circles.pop(i)
                    break
            j += 1
        i += 1
next_fruit_index = random.randint(0, len(FRUIT_COLOR_SIZE)-1)
next_fruit_index = PhysicsCircle((50, 100), next_fruit_index)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            circles.append(PhysicsCircle(event.pos, random.randint(0, len(FRUIT_COLOR_SIZE)-1)))

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
    handle_collisions(circles)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
