from .setting import *
import pymunk
import pygame
from pymunk.vec2d import Vec2d


def calc_mass(radius):
	# 質量の計算
	return radius ** 2


class PhysicsFruit(pymunk.Circle):
	def __init__(self, center, fruit_label):
		radius = FRUIT_INFO[fruit_label][1]
		m = calc_mass(radius)
		inertia = pymunk.moment_for_circle(m, 0, radius)
		body = pymunk.Body(m, inertia)
		super().__init__(body, radius)

		self.color = FRUIT_INFO[fruit_label][0]

		self.body.position = center
		self.body.velocity = Vec2d(0, 0)

		self.label = fruit_label
		self.elasticity = 0.01
		# self.surface_velocity = Vec2d(0.001, 0.001)
		self.friction = FRUIT_INFO[fruit_label][3]
		self.collision_type = 1

	def draw(self, screen):
		pygame.draw.circle(screen, self.color, (int(self.body.position[0]), int(self.body.position[1])), self.radius)


class Line(pymunk.Segment):
	def __init__(self, start, end, elasticity=0.2, friction=1.0, radius=LINE_WIDTH//2):
		body = pymunk.Body(body_type=pymunk.Body.STATIC)
		super().__init__(body, start, end, radius)
		self.elasticity = elasticity  # 弾性係数
		self.friction = friction  # 摩擦係数
		self.start = start
		self.end = end
		self.width = int(radius * 2)

	def draw(self, screen):
		pygame.draw.line(screen, (128, 128, 128), self.start, self.end, self.width)
