import pygame
import math
import random
from pygame.math import Vector2 as Vec2
from setting import *
import time
import threading
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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
    def __init__(self, start, end, line_width=5):
        self.start = Vec2(start)
        self.end   = Vec2(end)
        self.line_width = line_width

    def draw(self):
        pygame.draw.line(screen, (128, 128, 128), self.start, self.end, self.line_width)

    def closest(self, point):
        line_vec = self.end - self.start
        point_vec = point - self.start
        t = max(0, min(1, point_vec.dot(line_vec) / line_vec.length_squared()))
        return self.start + line_vec * t

class PhysicsCircle:
    def __init__(self, center, fruit_label,bottom_y=BOTTOM_Y,side_x=SIDE_X):
        self.pos = Vec2(center)
        self.fruit_label = fruit_label
        self.color  = FRUIT_COLOR_SIZE[fruit_label][0]
        self.radius = FRUIT_COLOR_SIZE[fruit_label][1]
        self.m = calculate_mass(self.radius)
        self.bottom_y = bottom_y # 箱からはみ出さないように
        self.side_x   = side_x   # 箱からはみ出さないように
        self.v = Vec2(0, 0)
        self.m = 1

    def update(self, delta):
        self.v.y += (GRAVITY*ONE_SECOND_FRAME* delta)/self.m  # Apply gravity
        self.pos += self.v * delta  # Update position based on velocity
        
        if self.pos.y+self.radius > self.bottom_y:
            self.pos.y = self.bottom_y - self.radius
            # Reflect the velocity and apply restitution
            self.v.y = -self.v.y * RESTITUTION
            
        if self.pos.x+self.radius < self.side_x[0]:
            self.pos.x = self.side_x[0] + self.radius
            self.v.x = -self.v.x * RESTITUTION
            
        if self.pos.x+self.radius > self.side_x[1]:
            self.pos.x = self.side_x[1] - self.radius
            self.v.x = -self.v.x * RESTITUTION
    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

# Create walls and circles list
walls = [
        Line(((SCREEN_WIDTH-BOX_WIDTH)//2, 240), ((SCREEN_WIDTH-BOX_WIDTH)//2,200+BOX_HEIGHT),BOX_LINE_WIDTH), # 左前壁
        Line((SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2,200+BOX_HEIGHT),BOX_LINE_WIDTH), # 右前壁
        Line(((SCREEN_WIDTH-BOX_WIDTH)//2, 200+BOX_HEIGHT), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 200+BOX_HEIGHT),BOX_LINE_WIDTH)   # 下壁
        ]
circles = []

def handle_collisions(circles):
    i = 0
    while i < len(circles):
        j = i + 1
        while j < len(circles):
            circle1, circle2 = circles[i], circles[j]
            if (circle1.pos - circle2.pos).length() < (circle1.radius + circle2.radius):
                if circle1.fruit_label == circle2.fruit_label and circle1.fruit_label < len(FRUIT_COLOR_SIZE) - 1:
                    new_label = circle1.fruit_label + 1
                    new_pos = (circle1.pos + circle2.pos) * 0.5
                    new_circle = PhysicsCircle(new_pos, new_label)
                    new_circle.v = (circle1.v * circle1.m + circle2.v * circle2.m) / (circle1.m + circle2.m)
                    circles.append(new_circle)
                    circles.pop(j)  # Remove second circle first
                    circles.pop(i)  # Remove first circle
                    i -= 1  # Adjust label after removal
                    break
                elif circle1.fruit_label == circle2.fruit_label and circle1.fruit_label == len(FRUIT_COLOR_SIZE) - 1:
                    circles.pop(j)
                    circles.pop(i)
                    break
            j += 1
        i += 1
        
def calculate_mass(radius):
    return radius ** 2
      
def create_next_fruit():
    next_fruit_label = random.randint(0, len(FRUIT_COLOR_SIZE)-7)
    next_fruit = PhysicsCircle((400.5, 30), next_fruit_label, BOTTOM_Y)
    return next_fruit, next_fruit_label

def resolve_collision(circle1, circle2):
    distance = circle1.pos.distance_to(circle2.pos)
    if distance < circle1.radius + circle2.radius:
        overlap = circle1.radius + circle2.radius - distance
        normal = (circle2.pos - circle1.pos).normalize()
        
        circle1.pos -= normal * (overlap / 2)
        circle2.pos += normal * (overlap / 2)
        
        # Resolve velocities
        relative_v = circle1.v - circle2.v
        v_along_normal = relative_v.dot(normal)
        if v_along_normal > 0:
            return
        
        restitution = 0.5  # Coefficient of restitution
        impulse_magnitude = -(1 + restitution) * v_along_normal
        impulse_magnitude /= (1 / circle1.m + 1 / circle2.m)
        
        impulse = impulse_magnitude * normal
        circle1.v -= impulse / circle1.m
        circle2.v += impulse / circle2.m

# 手につかんでいるフルーツ
now_fruit_label = random.randint(0, len(FRUIT_COLOR_SIZE)-7)
now_fruit = PhysicsCircle(pygame.mouse.get_pos(), now_fruit_label, BOTTOM_Y)

next_fruit, next_fruit_label = create_next_fruit()
print()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if CURSOR_BOUND_MIN_X < event.pos[0] < CURSOR_BOUND_MAX_X and event.pos[1] < 200+40:
                circles.append(PhysicsCircle(event.pos, now_fruit_label, BOTTOM_Y))
                print(event.pos)
                now_fruit_label = next_fruit_label
                now_fruit = PhysicsCircle(pygame.mouse.get_pos(), now_fruit_label, BOTTOM_Y)
                next_fruit, next_fruit_label = create_next_fruit()
                print(f"落とす果物:{now_fruit_label}, 次に来る果物{next_fruit_label}")
            
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
    screen.fill(BACKGROUND_COLOR)
    
    # Check for collisions
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            resolve_collision(circles[i], circles[j])
            
    # Draw Bax
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MIN_X,200+BOX_HEIGHT),5) # 左後壁
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MAX_X,200), (CURSOR_BOUND_MAX_X,200+BOX_HEIGHT),5) # 左後壁
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MAX_X,200),5) # 上後壁
    pygame.draw.line(screen, (200, 200, 200),(SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MAX_X,200),5)
    pygame.draw.line(screen, (200, 200, 200),((SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MIN_X,200),5)
    pygame.draw.line(screen, (128, 128, 128), ((SCREEN_WIDTH-BOX_WIDTH)//2, 200+40), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 200+40),5) # 上前壁
    now_fruit.pos = Vec2(pygame.mouse.get_pos())
    now_fruit.draw()
    next_fruit.draw()
    for circle in circles:
        circle.draw()
    for wall in walls:
        wall.draw()
    handle_collisions(circles)
    
    pygame.display.flip()
    clock.tick(60)
    
pygame.quit()
