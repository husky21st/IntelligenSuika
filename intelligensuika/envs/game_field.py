import pygame
import math
import random
from pygame.math import Vector2 as Vec2
from setting import *
import time

import pymunk
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
    return radius **2

def create_next_fruit():
    next_fruit_label = random.randint(1, len(FRUIT_INFO)-7)
    next_fruit = PhysicsCircle((400.5, 30), next_fruit_label)
    return next_fruit, next_fruit_label

def merge_fruits(arbiter, space, _):
    a, b = arbiter.shapes
    fruit1, fruit2 = a.body, b.body
    if fruit1.label == fruit2.label:
        mid_point = (fruit1.position + fruit2.position) / 2
        mid_v     = (fruit1.velocity + fruit2.velocity) / 2
        space.remove(a.body, a)
        space.remove(b.body, b)
        # circlesリストから削除
        circles.remove(fruit1.data)
        circles.remove(fruit2.data)
        if fruit1.label < (len(FRUIT_INFO)-1):
            new_label = fruit1.label + 1            
            circle = PhysicsCircle(mid_point, new_label)
            circle.velocity = mid_v
            circles.append(circle)
            space.add(circle.body,circle.shape)
        return False

    return True

def check_game_over(circles):
    global running
    for circle in circles:
        if circle.body.position.y + circle.radius < 240:
            print("Game Over")
            running = False
            return False
    return True
def convert_position(x):
    x = (x + 1)*0.5 # -1~1 -> 0~1
    x = x * (CURSOR_BOUND_MAX_X - CURSOR_BOUND_MIN_X) + CURSOR_BOUND_MIN_X
    return int(x)
# 衝突ハンドラの設定
handler = space.add_collision_handler(1, 1)
handler.begin = merge_fruits

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
    def __init__(self, center, fruit_label,bottom_y=BOTTOM_Y,side_x=SIDE_X):
        self.color  = FRUIT_INFO[fruit_label][0]
        self.radius = FRUIT_INFO[fruit_label][1]
        self.bottom_y = bottom_y
        self.side_x   = side_x
        self.m = calculate_mass(self.radius)

        inertia = pymunk.moment_for_circle(self.m, 0, self.radius)
        body    = pymunk.Body(self.m, inertia)
        body.position = center
        body.velocity = (0,0)
        body.label    = fruit_label
        body.data     = self
        shape = pymunk.Circle(body, self.radius)
        shape.elasticity = FRUIT_INFO[fruit_label][2]
        shape.friction   = FRUIT_INFO[fruit_label][3] 
        shape.collision_type = 1
        self.shape,self.body = shape, body
    
    def update(self):
        if self.body.position.y+self.radius > self.bottom_y+10:
            self.body.position = self.body.position.x, self.bottom_y - self.radius
            self.body.velocity = self.body.velocity.x, self.body.velocity.y * RESTITUTION
            
        if self.body.position.x+self.radius < self.side_x[0]-10:
            self.body.position = self.side_x[0] + self.radius, self.body.position.y
            self.body.velocity = -self.body.velocity.x * RESTITUTION, self.body.velocity.y
            
        if self.body.position.x+self.radius > self.side_x[1]:
            self.body.position = self.side_x[1] - self.radius, self.body.position.y
            self.body.velocity = -self.body.velocity.x * RESTITUTION, self.body.velocity.y
      
    def mouse_handle(self):
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
now_fruit_label = random.randint(1, len(FRUIT_INFO)-7)
now_fruit       = PhysicsCircle(pygame.mouse.get_pos(), now_fruit_label)

next_fruit, next_fruit_label = create_next_fruit()

circles = []  

# タイマーイベントの設定
DROP_FRUIT_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(DROP_FRUIT_EVENT, 1000)  # 5秒ごとにイベントを発生

def drop_fruit(x,now_fruit_label,next_fruit_label):
    y = 180 # 固定
    
    circle = PhysicsCircle((x, y), now_fruit_label)
    space.add(circle.body, circle.shape)
    circles.append(circle)
    now_fruit_label = next_fruit_label
    # now_fruit = PhysicsCircle(x, y, now_fruit_label)
    next_fruit, next_fruit_label = create_next_fruit()
    print(f"落とす果物:{now_fruit_label}, 次に来る果物{next_fruit_label}")
    
    return now_fruit,now_fruit_label,next_fruit,next_fruit_label

# メインループ
while running:
    # now_fruit.mouse_handle()
    for circle in circles:
        circle.update()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == DROP_FRUIT_EVENT and check_game_over(circles):            
            x = random.random()*2 - 1 # -1~1の乱数生成
            print(f"乱数x: {x}")
            x = convert_position(x)
            now_fruit,now_fruit_label,next_fruit,next_fruit_label = drop_fruit(x,now_fruit_label,next_fruit_label)
        # elif event.type == pygame.MOUSEBUTTONDOWN:
        #     x, y = event.pos
        #     if CURSOR_BOUND_MIN_X < x < CURSOR_BOUND_MAX_X and y < 200:
        #         circle = PhysicsCircle((x, y), now_fruit_label)
        #         space.add(circle.body, circle.shape)
        #         circles.append(circle)
        #         now_fruit_label = next_fruit_label
        #         now_fruit = PhysicsCircle(pygame.mouse.get_pos(), now_fruit_label)
        #         next_fruit, next_fruit_label = create_next_fruit()
        #         print(f"落とす果物:{now_fruit_label}, 次に来る果物{next_fruit_label}")
    # 物理シミュレーションを進める
    space.step(1 / 50.0)

    # 画面をクリア
    screen.fill((255, 255, 255))
    
    # Draw Bax
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MIN_X,200+BOX_HEIGHT),5) # 左後壁
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MAX_X,200), (CURSOR_BOUND_MAX_X,200+BOX_HEIGHT),5) # 左後壁
    pygame.draw.line(screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MAX_X,200),5)            # 上後壁
    pygame.draw.line(screen, (200, 200, 200),(SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MAX_X,200),5)
    pygame.draw.line(screen, (200, 200, 200),((SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MIN_X,200),5)
    # now_fruit.draw()
    next_fruit.draw()

    for circle in circles:
        circle.draw()
    for wall in walls:
        wall.draw()
    
    pygame.draw.line(screen, (128, 128, 128),((SCREEN_WIDTH-BOX_WIDTH)//2, 240),(SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240),5)
    # 画面を更新
    pygame.display.flip()

    # 60FPSで実行
    clock.tick(60)

pygame.quit()
