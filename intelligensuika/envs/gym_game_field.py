import gym
from gym import spaces
import pymunk
from pymunk.vec2d import Vec2d
from typing import Optional

import pygame
import numpy as np
import random
from setting import *
import time

class PhysicsCircle:
    def __init__(self, center, fruit_label,bottom_y=BOTTOM_Y,side_x=SIDE_X):
        self.color  = FRUIT_INFO[fruit_label][0]
        self.radius = FRUIT_INFO[fruit_label][1]
        self.bottom_y = bottom_y
        self.side_x   = side_x
        self.m = self.calc_mass(self.radius)

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

    def calc_mass(self,radius):
        # 質量の計算
        return radius**2
    
    def draw(self,screen):
        pygame.draw.circle(screen, self.color, (int(self.body.position[0]), int(self.body.position[1])), self.radius)

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
        
    def draw(self,screen):
        pygame.draw.line(screen, (128, 128, 128), self.start, self.end, self.width)
        

def convert_position(x):
    x = x[0]
    x = (x + 1)*0.5 # -1~1 -> 0~1
    x = x * (CURSOR_BOUND_MAX_X - CURSOR_BOUND_MIN_X) + CURSOR_BOUND_MIN_X
    return int(x)

class SuikaEnv(gym.Env):
    metadate ={
        'render.modes': ['human', 'rgb_array'],
        "render_fps": FRAMES_PER_SECOND # 1秒間のフレーム数
    }
    
    def __init__(self,render_mode: Optional[str] = None, g=10.0):
        self.render_mode = render_mode
        self.fruit_info = FRUIT_INFO
        self.fruit_box  = []
        
        self.wait_frames  = WAIT_FRAMES
        self.frame_count  = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)      # -1~1の値を受け取る
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) # 3つの値を返す
        
    def step(self,action):
        # actionは-1~1の値
        # actionを受けて、次の状態,報酬,エピソード終了判定(Game Overかどうか)を返す.
        print(f"action:{action}")
        action = convert_position(action)
        self.drop_fruit(action)
        if self.render_mode == 'human':
            # 指定フレーム数で待機
            while self.frame_count < self.wait_frames:
                self.render()
                pygame.display.flip()
                self.clock.tick(60)  # 60FPSで動作
                self.frame_count += 1
            self.frame_count = 0 
        else: # mode in "rgb_array"
            time.sleep(3) # 仮の待機時間
        cost = 0 # 仮のコスト(報酬？)
        return self._get_obs(), -cost, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # pymunkの初期化
        self.space         = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        
        # 衝突ハンドラの設定
        self.handler = self.space.add_collision_handler(1, 1)
        self.handler.begin = self.merge_fruits
        # 壁の初期化
        self.walls = []
        self.walls.append(Line(((SCREEN_WIDTH-BOX_WIDTH)//2, 240), ((SCREEN_WIDTH-BOX_WIDTH)//2,200+BOX_HEIGHT),WALL_ELASTICITY,WALL_FRICTION))                           # 左前壁
        self.walls.append(Line((SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2,200+BOX_HEIGHT),WALL_ELASTICITY,WALL_FRICTION)) # 右前壁
        self.walls.append(Line(((SCREEN_WIDTH-BOX_WIDTH)//2, 200+BOX_HEIGHT), (SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 200+BOX_HEIGHT),WALL_ELASTICITY,WALL_FRICTION))  # 下壁
        for wall in self.walls:
            self.space.add(wall.body, wall.shape)
        
        # 現在の果物の初期化
        self.now_fruit_label = random.randint(1, len(FRUIT_INFO)-7)
        # self.now_fruit       = PhysicsCircle((400.5, 30), self.now_fruit_label)
        
        # 次の果物の初期化
        self.next_fruit, self.next_fruit_label = self.create_next_fruit()
        if self.render_mode == 'human':
            self.screen = None
            self.clock  = None
            self.render()
        return self._get_obs(), {}
        
    def _get_obs(self):
        # 仮) 現在の果物のラベル, 次の果物のラベル, 現在の箱の状態(果物の位置,ラベル)
        obs = []
        for fruit in self.fruit_box:
            x = ((fruit.body.position.x-((SCREEN_WIDTH-BOX_WIDTH)//2)) / BOX_WIDTH)*2 -1
            y = 1-(fruit.body.position.y - 200) / BOX_HEIGHT
            obs.append([x,y,fruit.body.label])
        return obs
    
    def merge_fruits(self,arbiter, space, _):
        a, b = arbiter.shapes
        fruit1, fruit2 = a.body, b.body
        if fruit1.label == fruit2.label:
            mid_point = (fruit1.position + fruit2.position) / 2
            mid_v     = (fruit1.velocity + fruit2.velocity) / 2
            self.space.remove(a.body, a)
            self.space.remove(b.body, b)
            # fruit_boxリストから削除
            self.fruit_box.remove(fruit1.data)
            self.fruit_box.remove(fruit2.data)
            if fruit1.label < (len(FRUIT_INFO)-1):
                new_label = fruit1.label + 1            
                fruit = PhysicsCircle(mid_point, new_label)
                fruit.velocity = mid_v
                self.fruit_box.append(fruit)
                self.space.add(fruit.body,fruit.shape)
            return False
        return True
    def create_next_fruit(self):
        next_fruit_label = random.randint(1, len(FRUIT_INFO)-7)
        next_fruit = PhysicsCircle((400.5, 30), next_fruit_label)
        return next_fruit, next_fruit_label
    
    def check_game_over():
        # ゲームオーバーの判定
        pass
    
    def drop_fruit(self,x):
        y = 180 # 固定
        
        fruit = PhysicsCircle((x, y), self.now_fruit_label)
        self.space.add(fruit.body, fruit.shape)
        self.fruit_box.append(fruit)
        self.now_fruit_label = self.next_fruit_label
        self.next_fruit, self.next_fruit_label = self.create_next_fruit()
        print(f"落とす果物:{self.now_fruit_label}, 次に来る果物{self.next_fruit_label}")
        return
        
    # 描写に関わる関数
    def render(self, mode='human'):
        # pygameの初期化
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            else: # mode in "rgb_array"
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock  = pygame.time.Clock()
            
        self.space.step(1 / PYMUNK_FPS) # 物理シミュレーションの更新
        
        self.screen.fill((255,255,255))
        # 箱の描写 ()
        pygame.draw.line(self.screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MIN_X,200+BOX_HEIGHT),5) # 左後壁
        pygame.draw.line(self.screen, (200, 200, 200), (CURSOR_BOUND_MAX_X,200), (CURSOR_BOUND_MAX_X,200+BOX_HEIGHT),5) # 左後壁
        pygame.draw.line(self.screen, (200, 200, 200), (CURSOR_BOUND_MIN_X,200), (CURSOR_BOUND_MAX_X,200),5)            # 上後壁
        pygame.draw.line(self.screen, (200, 200, 200),(SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MAX_X,200),5)
        pygame.draw.line(self.screen, (200, 200, 200),((SCREEN_WIDTH-BOX_WIDTH)//2, 240),(CURSOR_BOUND_MIN_X,200),5)
        self.next_fruit.draw(self.screen)
        for fruit in self.fruit_box:
            fruit.draw(self.screen)
        for wall in self.walls:
            wall.draw(self.screen)
        pygame.draw.line(self.screen, (128, 128, 128),((SCREEN_WIDTH-BOX_WIDTH)//2, 240),(SCREEN_WIDTH-(SCREEN_WIDTH-BOX_WIDTH)//2, 240),5) 
              
        if self.render_mode == 'human':
            self.clock.tick(self.metadate["render_fps"])
            pygame.display.flip()
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False



import gym
import numpy as np

# 環境の作成
env = SuikaEnv(render_mode='human')

# 環境をリセットして初期状態を取得
state = env.reset()

# 何ステップかのシミュレーション
for _ in range(1000):
    # ランダムなアクションを生成
    action = env.action_space.sample()
    print(action)
    
    # アクションを環境に適用し、次の状態と報酬、終了フラグを取得
    state, reward, done, info = env.step(action)
    print(state, reward, done)
    
    # 状態を視覚的に表示
    env.render()
    
    # エピソードが終了したらループを抜ける
    if done:
        break

# 環境を閉じる
env.close()
