import pygame
import random
import numpy as np
from util import load_sprite

colors = ['red','green']
DOWN = 0
UP = 1

class Pipes():
    def __init__(self,color=None,pos=(0,0)):
        self.x,self.y = pos
        self.color = color
        self.gap = 100
        self.offset = self.gap
        self.high_height = -250
        self.lower_heght = -100
        self.speed = 7
        self.y = np.random.uniform(low=self.high_height,high=self.lower_heght)
        self.load_pipe_sprites()

    def reset(self,x):
        self.y = np.random.uniform(low=self.high_height,high=self.lower_heght)
        self.x = x

    def load_pipe_sprites(self):
        if not self.color:
            self.color = colors[random.randrange(0,len(colors))]
        else:
            self.color = self.color

        self.sprites = (
            load_sprite(f'pipe-{self.color}'),
            pygame.transform.flip(load_sprite(f'pipe-{self.color}'),False,True)
        )

        self.width, self.height = self.sprites[0].get_size()

    def update(self):
        self.x -= self.speed
        if(self.x + self.sprites[0].get_size()[0] <= 0):
            self.color = colors[random.randrange(0,len(colors))]
            self.x = 288 + self.sprites[0].get_size()[0]
            self.y = np.random.uniform(low=self.high_height,high=self.lower_heght)

    def draw(self,screen):
        screen.blit(self.sprites[UP], (self.x,self.y))
        screen.blit(self.sprites[DOWN], (self.x,self.y + self.sprites[0].get_size()[1] + self.gap))

    def get_hitboxes(self):
        return [self.sprites[0].get_rect(), self.sprites[1].get_rect()]