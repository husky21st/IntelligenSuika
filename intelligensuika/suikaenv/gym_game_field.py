import gym
from gym import spaces
import pymunk
from pymunk.vec2d import Vec2d
from typing import Optional
import pygame
import numpy as np
import random
from .setting import *
from .object_utils import *
import sys

def convert_position(x):
    x = x[0]
    x = (x + 1)*0.5 # -1~1 -> 0~1
    x = x * (CURSOR_BOUND_MAX_X - CURSOR_BOUND_MIN_X) + CURSOR_BOUND_MIN_X
    return int(x)

class SuikaEnv(gym.Env):
    metadate ={
        'render.modes': ['human', 'rgb_array'],
        "_render_fps": FRAMES_PER_SECOND # 1秒間のフレーム数
    }
    def __init__(self):
        self.fruit_info   = FRUIT_INFO
        self.wait_frames  = WAIT_FRAMES
        self.frame_count  = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)      # -1~1の値を受け取る
        self.default_observation  = [[0.0,0.0, 0] for _ in range(MAX_FRUIT_NUM)]          # 現在の果物と次の果物のラベルと60個の果物の位置を初期化 size: (62×3)
        self.observation_space    = spaces.Box(low=-1, high=11, shape=(MAX_FRUIT_NUM,3), dtype=np.float32)
        # self.max_fruit_num = MAX_FRUIT_NUM
        self.reward        = 0
        self.reset()
                
    def step(self,action):
        # actionは-1~1の値
        # actionを受けて、次の状態,報酬,エピソード終了判定(Game Overかどうか)を返す.
        action = convert_position(action)
        self.drop_fruit(action)
        
        self.reward = 0
        self.merge_fruits_lsit = []
        # 指定フレーム数で待機
        while self.frame_count < self.wait_frames:
            for fruit in self.fruit_box:
                fruit.pos_check()
            self.update()
            if True:
                self.render()
                self.clock.tick(self.metadate["_render_fps"])
            self.frame_count += 1
        
        self.now_fruit_label = self.next_fruit_label
        self.next_fruit, self.next_fruit_label = self.create_next_fruit()
        self.frame_count = 0
        self.reward      = self.calc_reward()
        done             = self.check_game_over()
        return self._get_obs(), self.reward, done, {}
        # return self._get_obs(), self.reward, done,{}, {}

    # def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    def reset(self):
    #     super().reset(seed=seed)
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
        self.fruit_box   = []
        self.merge_fruits_lsit = [] # 1stepで結合した果物リスト
        self.clock  = pygame.time.Clock()
        return self._get_obs()
    

    def seed(self, seed=None):
        pass
    
    def _get_obs(self):
        # 仮) 現在の果物のラベル, 次の果物のラベル, 現在の箱の状態(果物の位置)
        # sort　大きい順
        # obs = [[0.0,0.0] for _ in range(self.max_fruit_num)]
        obs = []
        for i in range(0, len(self.fruit_box)):
            x = ((self.fruit_box[i].body.position.x-((SCREEN_WIDTH-BOX_WIDTH)//2)) / BOX_WIDTH)*2 -1
            y = 1-(self.fruit_box[i].body.position.y - 200) / BOX_HEIGHT
            label = self.fruit_box[i].body.label
            obs.append([y,x,label])
        obs = np.array(obs, dtype=np.float32)
        if len(obs) == 0:
            obs = np.array([[0.0,0.0, 0] for _ in range(MAX_FRUIT_NUM)], dtype=np.float32)
        else:
            obs = obs[np.argsort(obs[:,0])]
            tmp = [[0.0,0.0, 0] for _ in range(MAX_FRUIT_NUM-len(self.fruit_box))]
            tmp = np.array(tmp, dtype=np.float32)
            obs = np.vstack([obs,tmp])
        obs = np.vstack([obs,[self.now_fruit_label, self.next_fruit_label, self.count_box_fruits()]])
        return obs.flatten()
    
    def calc_reward(self):
        # listの中から最大のものを選ぶ
        if len(self.merge_fruits_lsit) == 0:
            return 0
        max_label = max(self.merge_fruits_lsit)
        reward = self.fruit_info[max_label][4]
        return reward
    
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
                self.merge_fruits_lsit.append(new_label)
            return False
        return True
    
    def count_box_fruits(self):
        return len(self.fruit_box)
        
    def create_next_fruit(self):
        next_fruit_label = random.randint(1, len(FRUIT_INFO)-7)
        next_fruit = PhysicsCircle((400.5, 30), next_fruit_label)
        return next_fruit, next_fruit_label
    
    def check_game_over(self):
        for fruit in self.fruit_box:
            if fruit.body.position.y + fruit.radius < GAME_OVER_LINE:
                self.reward = -1
                return True 
        return False

    def drop_fruit(self,x):
        y = 180 # 固定
        fruit = PhysicsCircle((x, y), self.now_fruit_label)
        self.space.add(fruit.body, fruit.shape)
        self.fruit_box.append(fruit)

    
    def update(self):
         self.space.step(1 / PYMUNK_FPS) # 物理シミュレーションの更新
    
    # 描写に関わる関数
    def render(self, mode='human', close=False):
        if self.screen is None:
            pygame.init()
            if mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            # else: # mode in "rgb_array"
            #     self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock  = pygame.time.Clock()
        self.space.step(1 / PYMUNK_FPS) # 物理シミュレーションの更新
        if mode == 'rgb_array':
            self.clock.tick(self.metadate["_render_fps"])  
            return np.array(pygame.surfarray.pixels3d(self.screen))
            
        elif mode == 'human':
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
            self.clock.tick(self.metadate["_render_fps"])
            pygame.display.flip()
            return None

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# import intelli_suika
# import gym
# import numpy as np
# env = gym.make('intelligencesuika-v0')
# env.mode = 'human'
# state = env.reset()

# # mode in "human" or "rgb_array"
# env = SuikaEnv(mode='human')
# # 環境をリセットして初期状態を取得
# state = env.reset()
# 何ステップかのシミュレーション
# for i in range(10):
#     print(f"episode:{i}")
#     done = False
#     state = env.reset()
#     while not done:
        
#         # ランダムなアクションを生成
#         action = env.action_space.sample()
#         # アクションを環境に適用し、次の状態と報酬、終了フラグを取得
#         state, reward, done, info = env.step(action)
#         print(reward, done)
#         env._render()
# # 環境を閉じる
# env.close()