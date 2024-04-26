import gym
from gym import spaces
import pymunk
from pymunk.vec2d import Vec2d
from typing import Optional
import pygame
import numpy as np
import random
from setting import *
from object_utils import *

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
        self.fruit_info  = FRUIT_INFO
        self.fruit_box   = []
        self.wait_frames  = WAIT_FRAMES
        self.frame_count  = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)      # -1~1の値を受け取る
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2つの値を返す
        self.default_observation  = [[0.0,0.0] for _ in range(MAX_FRUIT_NUM)]                     # 60個の果物の位置を初期化
        # self.max_fruit_num = MAX_FRUIT_NUM
        self.reward        = REWARD_DEFAULT
        self.total_reward  = 0
        
    def step(self,action):
        # actionは-1~1の値
        # actionを受けて、次の状態,報酬,エピソード終了判定(Game Overかどうか)を返す.
        print(f"action:{action}")
        action = convert_position(action)
        print(f"action:{action}")
        self.total_reward = 0 # 報酬の初期化
        self.drop_fruit(action)
        
        # 指定フレーム数で待機
        while self.frame_count < self.wait_frames:
            for fruit in self.fruit_box:
                fruit.update()
            self.render()
            self.clock.tick(self.metadate["render_fps"])
            self.frame_count += 1
        self.frame_count = 0
        done = self.check_game_over()
        return self._get_obs(), self.total_reward, done, {}
    
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
        self.next_fruit, self.next_fruit_label = self.create_next_fruit()
        self.screen = None
        self.clock  = None
        self.render()
        return self._get_obs(), {}
    
    def _get_obs(self):
        # 仮) 現在の果物のラベル, 次の果物のラベル, 現在の箱の状態(果物の位置)
        # obs = [[0.0,0.0] for _ in range(self.max_fruit_num)]
        obs = self.default_observation
        for i in range(len(self.fruit_box)):
            x = ((self.fruit_box[i].body.position.x-((SCREEN_WIDTH-BOX_WIDTH)//2)) / BOX_WIDTH)*2 -1
            y = 1-(self.fruit_box[i].body.position.y - 200) / BOX_HEIGHT
            obs[i] = [x,y]
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
            if fruit1.data in self.fruit_box:
                self.fruit_box.remove(fruit1.data)
            if fruit2.data in self.fruit_box:
                self.fruit_box.remove(fruit2.data)
            if fruit1.label < (len(FRUIT_INFO)-1):
                new_label = fruit1.label + 1
                fruit = PhysicsCircle(mid_point, new_label)
                fruit.velocity = mid_v
                self.fruit_box.append(fruit)
                self.space.add(fruit.body,fruit.shape)
                self.total_reward += self.reward
            return False
        return True
    
    def create_next_fruit(self):
        next_fruit_label = random.randint(1, len(FRUIT_INFO)-7)
        next_fruit = PhysicsCircle((400.5, 30), next_fruit_label)
        return next_fruit, next_fruit_label
    
    def check_game_over(self):
        for fruit in self.fruit_box:
            if fruit.body.position.y + fruit.radius < 240:
                return True
        return False
    
    def drop_fruit(self,x):
        y = 180 # 固定
        fruit = PhysicsCircle((x, y), self.now_fruit_label)
        self.space.add(fruit.body, fruit.shape)
        self.fruit_box.append(fruit)
        self.now_fruit_label = self.next_fruit_label
        self.next_fruit, self.next_fruit_label = self.create_next_fruit()
        print(f"落とす果物:{self.now_fruit_label}, 次に来る果物{self.next_fruit_label}")
    
    # 描写に関わる関数
    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            # else: # mode in "rgb_array"
            #     self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock  = pygame.time.Clock()
            
        self.space.step(1 / PYMUNK_FPS) # 物理シミュレーションの更新
        if self.render_mode == 'rgb_array':
            self.clock.tick(self.metadate["render_fps"])  
            
        elif self.render_mode == 'human':
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
            self.clock.tick(self.metadate["render_fps"])
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            
# import gym
# import numpy as np

# # mode in "human" or "rgb_array"
# env = SuikaEnv(render_mode='human')
# # 環境をリセットして初期状態を取得
# state = env.reset()
# # 何ステップかのシミュレーション
# for _ in range(1000):
#     # ランダムなアクションを生成
#     action = env.action_space.sample()
#     # アクションを環境に適用し、次の状態と報酬、終了フラグを取得
#     state, reward, done, info = env.step(action)
#     print(reward, done)
#     env.render()
#     if done:
#         break
# # 環境を閉じる
# env.close()