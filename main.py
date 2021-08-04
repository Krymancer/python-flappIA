import pygame
import random
import numpy as np
import copy

from game import Game
from player import Player
from pipes import Pipes
from nn import NeuralNetwork

pipe_colors = ['red','green']
pipe_color = pipe_colors[random.randrange(0,len(pipe_colors))]


total_population = 500
generation = 0
best_score = 0
all_birds = []
active_birds = []


def normalize_fitness(birds):
    sum = 0
    for bird in birds:
        sum += bird.score

    for bird in birds:
        bird.fitness = bird.score / sum

def pool_selection(birds):
    index = 0
    r = np.random.rand()
    while r > 0:
        r-= birds[index].fitness
        index += 1
    return birds[index - 1]

def generate(birds):
    new_birds = []
    for i in range(len(birds)):
        brain = pool_selection(birds).brain
        new_bird = Player(brain=brain)
        new_birds.append(new_bird)
    return new_birds

def next_generation():
    global generation
    global all_birds
    global active_birds

    generation += 1
    normalize_fitness(all_birds)
    active_birds = generate(all_birds)
    all_birds = copy.copy(active_birds)


def create_population():
    global all_birds
    global active_birds
    for x in range(total_population):
        bird = Player()
        all_birds.append(bird)
    
    active_birds = copy.copy(all_birds)

def reset(pipes):
    pipes[0].reset(288)
    pipes[1].reset(288 * 1.7)

def print_info(screen=None,font=None):
    global generation
    global best_score
    global active_birds
    generationSuf = font.render('Generation: ' + str(generation),True,pygame.Color('white'))
    bestScoreSuf = font.render('Best Score: ' + str(best_score),True,pygame.Color('white'))
    populationSuf = font.render('Population: ' + str(len(active_birds)),True,pygame.Color('white'))
    screen.blit(generationSuf, (10, 10 + 26))
    screen.blit(bestScoreSuf, (10, 10 + 26 * 2))
    screen.blit(populationSuf, (10, 10 + 26 * 3))

def main():
    global best_score
    global all_birds
    global active_birds
    
    game = Game(size=(288,512),name="Flappybird")
    game.load_assets()
    font = pygame.font.Font(None, 26)

    pipes = ( 
        Pipes(pos=(288,0),color=pipe_color),
        Pipes(pos=(288 * 1.7,0),color=pipe_color),
    )

    create_population()

    while game.running:
        game.clock.tick(30)
        # Handle Events
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                game.running = False

        if len(active_birds) == 0:
            reset(pipes)
            next_generation()

        for pipe in pipes:
            pipe.update()
            for bird in active_birds:
                if bird.collide(pipe):
                    bird.die()
                    active_birds.remove(bird)

        for bird in active_birds:
            bird.update()
            bird.think(pipes)


        game.draw_bg()
        for pipe in pipes:
            pipe.draw(screen=game.screen)

        for bird in active_birds:
            bird.draw(screen=game.screen)
            if bird.score > best_score:
                best_score = bird.score
        game.draw_fg()
        game.draw_fps()
        print_info(font=font,screen=game.screen)
        game.draw()

if __name__ == "__main__":
    main()