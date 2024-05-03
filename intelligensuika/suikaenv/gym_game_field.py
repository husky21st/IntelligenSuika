import gymnasium
from gymnasium import spaces
import pymunk
from typing import Optional
import pygame
import numpy as np
import random
from .setting import *
from .object_utils import PhysicsFruit, Line
from copy import copy

WALLS = [
	Line(((SCREEN_WIDTH - BOX_WIDTH) // 2 - LINE_WIDTH // 2, LINE_TOP_Y), ((SCREEN_WIDTH - BOX_WIDTH) // 2 - LINE_WIDTH // 2, BOX_BOTTOM_Y), WALL_ELASTICITY, WALL_FRICTION),  # 左前壁
	Line((SCREEN_WIDTH - (SCREEN_WIDTH - BOX_WIDTH) // 2 + LINE_WIDTH // 2, LINE_TOP_Y), (SCREEN_WIDTH - (SCREEN_WIDTH - BOX_WIDTH) // 2 + LINE_WIDTH // 2, BOX_BOTTOM_Y), WALL_ELASTICITY, WALL_FRICTION),  # 右前壁
	Line(((SCREEN_WIDTH - BOX_WIDTH) // 2, BOX_BOTTOM_Y + LINE_WIDTH // 2), (SCREEN_WIDTH - (SCREEN_WIDTH - BOX_WIDTH) // 2, BOX_BOTTOM_Y + LINE_WIDTH // 2), WALL_ELASTICITY, WALL_FRICTION),  # 下壁
	Line(((SCREEN_WIDTH - BOX_WIDTH) // 2 - LINE_WIDTH // 2, LINE_TOP_Y), (-200, LINE_TOP_Y), WALL_ELASTICITY, WALL_FRICTION, 1),  # 左壁の延長
	Line((SCREEN_WIDTH - (SCREEN_WIDTH - BOX_WIDTH) // 2 + LINE_WIDTH // 2, LINE_TOP_Y), (SCREEN_WIDTH+200, LINE_TOP_Y), WALL_ELASTICITY, WALL_FRICTION, 1),  # 右壁の延長
]


def create_new_label():
	return random.randint(1, len(FRUIT_INFO) - 7)


def convert_position(x: float):
	x = (x + 1) * 0.5  # -1~1 -> 0~1
	x = x * (CURSOR_BOUND_MAX_X - CURSOR_BOUND_MIN_X) + CURSOR_BOUND_MIN_X
	return x


class SuikaEnv(gymnasium.Env):
	metadata = {
		'render_modes': ['human', 'rgb_array'],
		"render_fps": FRAMES_PER_SECOND  # 1秒間のフレーム数
	}

	def __init__(self, render_mode: Optional[str] = None, skip_frame: Optional[int] = 1):
		self.render_mode = render_mode
		self.skip_frame = skip_frame
		self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # -1~1の値を受け取る
		self.observation_space = spaces.Box(low=-1, high=100, shape=(MAX_FRUIT_NUM + 1, 3), dtype=np.float32)
		self.reward = 0

		self.screen = None
		self.clock = None

		self.space = pymunk.Space(threaded=False)
		self.space.gravity = (0, GRAVITY)
		self.space.collision_bias = 0.001
		self.space.collision_persistence = 3
		self.space.damping = 0.5
		self.handler = self.space.add_collision_handler(1, 1)
		self.handler.begin = self.merge_fruits
		self.space.add(*[wall.body for wall in WALLS], *[wall for wall in WALLS])

		self.fruit_box = list()
		self.now_fruit_label = create_new_label()
		self.next_fruit_label = create_new_label()

		self.frame_count = 0
		self.max_label_merged = 0

	def step(self, action):
		# actionは-1~1の値
		# actionを受けて、次の状態,報酬,エピソード終了判定(Game Overかどうか)を返す.
		self.reward = 0
		obs = np.zeros(self.observation_space.shape, dtype=np.float32)
		done = False
		if self.frame_count % self.skip_frame == 0:
			action = convert_position(action)
			self.drop_fruit(action)
			self.now_fruit_label = self.next_fruit_label
			self.next_fruit_label = create_new_label()
			self.max_label_merged = 0
		elif (self.frame_count + 1) % self.skip_frame == 0:
			obs = self._get_obs()
			done = self.check_game_over(obs)
			self.reward = FRUIT_INFO[self.max_label_merged][4]
		self.update()
		self.frame_count += 1
		if self.render_mode == "human":
			self.render()
		return obs, self.reward, done, False, {}

	def reset(self, seed=None, options=None):
		super().reset(seed=seed, options=options)
		self.screen = None
		self.clock = None
		# 現在の果物の初期化
		self.remove_all_fruit()
		self.now_fruit_label = create_new_label()
		self.next_fruit_label = create_new_label()

		self.frame_count = 0
		self.max_label_merged = 0
		if self.render_mode == "human":
			self.render()
		return self._get_obs(), {}

	def _get_obs(self):
		# sort　大きい順
		obs = np.zeros((MAX_FRUIT_NUM + 1, 3), dtype=np.float32)
		for i, _fruit in enumerate(self.fruit_box):
			obs[i, 1] = ((_fruit.body.position.x - ((SCREEN_WIDTH - BOX_WIDTH) // 2)) / BOX_WIDTH) * 2 - 1
			obs[i, 0] = 1 - (_fruit.body.position.y - _fruit.radius - BOX_TOP_Y) / BOX_HEIGHT
			obs[i, 2] = _fruit.label
		obs = obs[np.argsort(obs[:, 0])[::-1]]
		obs[-1] = np.array([self.now_fruit_label, self.next_fruit_label, self.count_box_fruits()])
		return obs.astype(np.float32)

	def merge_fruits(self, arbiter, space, _):
		a, b = arbiter.shapes
		if a.label == b.label and a in self.fruit_box and b in self.fruit_box:
			merged_label = a.label
			self.remove_fruit(a)
			self.remove_fruit(b)
			self.max_label_merged = max(self.max_label_merged, merged_label)
			mid_point = (a.body.position + b.body.position) / 2
			if merged_label < (len(FRUIT_INFO) - 1):
				new_label = merged_label + 1
				self.add_fruit(mid_point, new_label)
			return False
		return True

	def count_box_fruits(self):
		return len(self.fruit_box)

	def check_game_over(self, obs):
		if self.count_box_fruits() >= MAX_FRUIT_NUM - 1 or obs[0, 0] > GAME_OVER_LINE:
			self.reward = -1
			return True
		return False

	def add_fruit(self, point, label):
		new_fruit = PhysicsFruit(point, label)
		self.fruit_box.append(new_fruit)
		self.space.add(new_fruit.body, new_fruit)

	def remove_fruit(self, delete_fruit):
		self.space.remove(delete_fruit.body, delete_fruit)
		self.fruit_box.remove(delete_fruit)
		del delete_fruit

	def remove_all_fruit(self):
		delete_box = copy(self.fruit_box)
		for del_fruit in delete_box:
			self.remove_fruit(del_fruit)
		del delete_box, self.fruit_box
		self.fruit_box = list()

	def drop_fruit(self, x):
		self.add_fruit((x, CURSOR_Y), self.now_fruit_label)

	def update(self):
		self.space.step(1 / PYMUNK_FPS)  # 物理シミュレーションの更新

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
			self.clock = pygame.time.Clock()
		if mode == 'rgb_array':
			self.clock.tick(self.metadata["render_fps"])
			return np.array(pygame.surfarray.pixels3d(self.screen))

		elif mode == 'human':
			self.screen.fill((255, 255, 255))
			# 箱の描写 ()
			# pygame.draw.line(self.screen, (200, 200, 200), (CURSOR_BOUND_MIN_X, 200),
			# 				 (CURSOR_BOUND_MIN_X, 200 + BOX_HEIGHT), 5)  # 左後壁
			# pygame.draw.line(self.screen, (200, 200, 200), (CURSOR_BOUND_MAX_X, 200),
			# 				 (CURSOR_BOUND_MAX_X, 200 + BOX_HEIGHT), 5)  # 左後壁
			# pygame.draw.line(self.screen, (200, 200, 200), (CURSOR_BOUND_MIN_X, 200), (CURSOR_BOUND_MAX_X, 200),
			# 				 5)  # 上後壁
			# pygame.draw.line(self.screen, (200, 200, 200), (SCREEN_WIDTH - (SCREEN_WIDTH - BOX_WIDTH) // 2, 240),
			# 				 (CURSOR_BOUND_MAX_X, 200), 5)
			# pygame.draw.line(self.screen, (200, 200, 200), ((SCREEN_WIDTH - BOX_WIDTH) // 2, 240),
			# 				 (CURSOR_BOUND_MIN_X, 200), 5)
			# self.next_fruit.draw(self.screen)
			for fruit in self.fruit_box:
				fruit.draw(self.screen)
			for wall in WALLS:
				wall.draw(self.screen)
			# pygame.draw.line(self.screen, (128, 128, 128), ((SCREEN_WIDTH - BOX_WIDTH) // 2, 240),
			# 				 (SCREEN_WIDTH - (SCREEN_WIDTH - BOX_WIDTH) // 2, 240), 5)
			self.clock.tick(self.metadata["render_fps"])
			pygame.display.flip()
			return None

	def close(self):
		if self.screen is not None:
			pygame.display.quit()
			pygame.quit()
