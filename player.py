import pygame
import random
import numpy as np

import math

from util import load_sprite, load_sound
from nn import NeuralNetwork

colors = ['red','blue','yellow']

UP = 0
MID = 1
DOWN = 2

def randomZero_One():
    u = 0
    v = 0
    while u == 0:
        u = np.random.rand() #Converting [0,1) to (0,1)
    while v == 0:
        v = np.random.rand()

    return math.sqrt( -2.0 * math.log( u ) ) * math.cos( 2.0 * math.pi * v );

def mutate_function(x):
    if np.random.rand() < 0.1:
        offset = randomZero_One() * 0.5
        newx = x + offset
        return newx
    else:
        return x

def map(value, low1, high1, low2, high2):
    return low2 + (high2 - low2) * (value - low1) / (high1 - low1)

class Player(pygame.sprite.Sprite):
    def __init__(self,color=None,pos=(0,0),brain=None):
        pygame.sprite.Sprite.__init__(self)
        self.x,self.y = pos
        self.color = color
        self.playerAnimation = [UP, MID, DOWN, MID]
        self.currentAnimationSate = 0
        self.framesUntilNextAnimaition = 5
        self.frames = 0
        self.jumped = False
        self.gravity = 1
        self.velocity = 0
        self.lift = -5
        self.alive = True
        self.rotate = 0
        self.score = 0
        self.fitness = 0
        self.load_player_sprite()

        if brain:
            self.brain = brain
            self.mutate()
        else:
            self.brain = NeuralNetwork(input_nodes=5,hidden_nodes=8,output_nodes=2)

    def mutate(self):
        self.brain.mutate(mutate_function)

    def load_player_sprite(self):
        if not self.color:
            color = colors[random.randrange(0,len(colors))]
        else:
            color = self.color

        base_names = (
            f'{color}bird-upflap',
            f'{color}bird-midflap',
            f'{color}bird-downflap',
        )

        self.sprites = (
            load_sprite(base_names[0]),
            load_sprite(base_names[1]),
            load_sprite(base_names[2]),
        )

        self.width, self.height = self.sprites[0].get_size()

        self.die_sound = load_sound('die')
        self.hit_sound = load_sound('hit')

    def die(self):
        if self.alive:
            self.hit_sound.play()
            self.die_sound.play()
            self.alive = False

    def out_bounds(self):
        return ((self.y >= 512 - 112) or (self.y < -self.sprites[0].get_size()[1])) and self.alive

    def update(self):
        if self.out_bounds():
            self.die()

        if self.rotate > 0:
            self.rotate -= 20

        self.velocity += self.gravity
        self.y += self.velocity
        self.jumped = False
        self.score += 1
    
    def collide(self,pipe):
        return self.x + self.width > pipe.x and (self.y < pipe.y + pipe.height or self.y + self.height > pipe.y + pipe.height + pipe.offset)

    def jump(self):
        if not self.jumped and self.alive:
            self.rotate = 45
            self.jumped = True
            self.velocity = self.lift

    def draw(self,screen):
        self.frames = self.frames + 1;
        if self.frames > self.framesUntilNextAnimaition:
            self.frames = 0
            self.currentAnimationSate = (self.currentAnimationSate+1) % len(self.playerAnimation)

        currentSprite = self.sprites[self.playerAnimation[self.currentAnimationSate]]
        currentSprite = pygame.transform.rotate(currentSprite,self.rotate)
        screen.blit(currentSprite, (self.x,self.y))

    def think(self,pipes):
        pipe = pipes[0] if pipes[0].x < pipes[1].x else pipes[1]

        inputs = []
        # x position of closest pipe
        inputs.append(map(pipe.x, self.x, 288, 0, 1))
        # top of closest pipe opening
        inputs.append(map(pipe.y + pipe.height, 0, 512, 0, 1))
        # bottom of closest pipe opening
        inputs.append(map(pipe.y + pipe.height + pipe.offset, 0, 512, 0, 1))
        # bird's y tion
        inputs.append(map(self.y, 0, 512, 0, 1))
        # pipe's x tion
        inputs.append(map(pipe.x,0, 288,0,1))
        # Get the outputs from the network
        action = self.brain.predict(inputs)

        if (action[1] > action[0]):
            self.jump()

    def copy(self):
        return self