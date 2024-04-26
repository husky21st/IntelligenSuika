from setting import *
import pymunk
import pygame
import random
import numpy as np


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
        