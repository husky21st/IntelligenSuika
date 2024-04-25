import gym
from gym import spaces
import pymunk
from pymunk.vec2d import Vec2d
from typing import Optional

import pygame
import numpy as np
import random
from setting import *

class SuikaEnv(gym.Env):
    metadate ={
        'render.modes': ['human', 'rgb_array'],
        "render_fps": 30
    }
    
    def __init__(self,render_mode: Optional[str] = None, g=10.0):
        self.fruit_info = FRUIT_INFO
        self.fruit_box  = []
        
    def step(self,action):
        # actionは-1~1の値
        # actionを受けて、次の状態,報酬,エピソード終了判定(Game Overかどうか)を返す.
        pass
    def reset(self):
        # ゲームの初期化: 箱の中にある果物をなくす
        pass
        
    def _get_obs(self):
        # 仮) 現在の果物のラベル, 次の果物のラベル, 現在の箱の状態(果物の位置,ラベル)
        obs = []
        for fruit in self.fruit_box:
            x = ((fruit.body.position.x-((SCREEN_WIDTH-BOX_WIDTH)//2)) / BOX_WIDTH)*2 -1
            y = 1-(fruit.body.position.y - 200) / BOX_HEIGHT
            obs.append([x,y,fruit.body.label])
        return obs
    
    def merge_fruits():
        # 同じ果物の結合
        pass
    
    def create_next_fruit():
        # 次の果物を作成
        pass
    
    def calc_mass():
        # 質量の計算
        pass
    
    def check_game_over():
        # ゲームオーバーの判定
        pass