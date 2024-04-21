import pygame
import math
import random
from pygame.math import Vector2 as Vec2
from setting import *
import time
import threading
import pymunk
import pymunk.pygame_util
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
running = True

# Pymunkの初期化
space = pymunk.Space()
space.gravity = (0, GRAVITY)  # 重力を設定 (m/s²)

def normalized(v):
    if v.length() == 0:
        return Vec2(0, 0)
    return v.normalize()


def calculate_mass(radius):
    return radius ** 2

def create_next_fruit():
    next_fruit_label = random.randint(0, len(FRUIT_INFO)-7)
    next_fruit = PhysicsCircle((400.5, 30), next_fruit_label)
    return next_fruit, next_fruit_label

class Line:
    def __init__(self, start, end,elasticity=0.2,friction=1.0, radius=2.5,):
        body  = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, start, end, radius)
        shape.elasticity = elasticity # 弾性係数
        shape.friction   = friction   # 摩擦係数
        self.shape, self.body = shape, body
        self.start = start
        self.end   = end
        self.width = int(radius*2)
    def draw(self):
        pygame.draw.line(screen, (128, 128, 128), self.start, self.end, self.width)
        

class PhysicsCircle:
    def __init__(self, center, fruit_label):
        self.fruit_label = fruit_label
        self.color  = FRUIT_INFO[fruit_label][0]
        self.radius = FRUIT_INFO[fruit_label][1]
        self.m = calculate_mass(self.radius)

        inertia = pymunk.moment_for_circle(self.m, 0, self.radius)
        body    = pymunk.Body(self.m, inertia)
        body.position = center
        shape = pymunk.Circle(body, self.radius)
        shape.elasticity = FRUIT_INFO[fruit_label][2]
        shape.friction   = FRUIT_INFO[fruit_label][3] 
        shape.collision_type = 1
        self.shape,self.body = shape, body
        
    def update(self):
        self.body.position = pygame.mouse.get_pos()

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.body.position[0]), int(self.body.position[1])), self.radius)
walls = []
walls.append(Line(((SCREEN_WIDTH-BOX_WIDTH)//2, 240), ((SCREEN_WIDTH-BOX_WIDTH)//2,200+BOX_HEIGHT),WALL_ELASTICITY,WALL_FRICTION))                           # 左前壁
walls.append(Line((SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2,200+BOX_HEIGHT),WALL_ELASTICITY,WALL_FRICTION)) # 右前壁
walls.append(Line(((SCREEN_WIDTH-BOX_WIDTH)//2, 200+BOX_HEIGHT), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 200+BOX_HEIGHT),WALL_ELASTICITY,WALL_FRICTION))  # 下壁
for wall in walls:
    space.add(wall.body, wall.shape)
    
# 手につかんでいるフルーツ
now_fruit_label = random.randint(0, len(FRUIT_INFO)-7)
now_fruit       = PhysicsCircle(pygame.mouse.get_pos(), now_fruit_label)

next_fruit, next_fruit_label = create_next_fruit()

circles = []        
# メインループ
while running:
    now_fruit.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if CURSOR_BOUND_MIN_X < x < CURSOR_BOUND_MAX_X and y < 200:
                circle = PhysicsCircle((x, y), now_fruit_label)
                space.add(circle.body, circle.shape)
                circles.append(circle)
                now_fruit_label = next_fruit_label
                now_fruit = PhysicsCircle(pygame.mouse.get_pos(), now_fruit_label)
                next_fruit, next_fruit_label = create_next_fruit()
                print(f"落とす果物:{now_fruit_label}, 次に来る果物{next_fruit_label}")
    # 物理シミュレーションを進める
    space.step(1 / 50.0)

    # 画面をクリア
    screen.fill((255, 255, 255))
    
    # Draw Bax
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MIN_X,200+BOX_HEIGHT),5) # 左後壁
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MAX_X,200), (CURSOR_BOUND_MAX_X,200+BOX_HEIGHT),5) # 左後壁
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MAX_X,200),5) # 上後壁
    pygame.draw.line(screen, (200, 200, 200),(SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MAX_X,200),5)
    pygame.draw.line(screen, (200, 200, 200),((SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MIN_X,200),5)

    now_fruit.draw()
    next_fruit.draw()

    for circle in circles:
        circle.draw()
    for wall in walls:
        wall.draw()
    # 画面を更新
    pygame.display.flip()

    # 60FPSで実行
    clock.tick(60)

pygame.quit()
