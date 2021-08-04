import pygame
import random

from util import load_sprite

#Radnom
backgrounds = ['background-day','background-night']

class Game():
    def __init__(self,size=(480,640),name="Pygame Window"):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.running = True
        self.width = size[0]
        self.height = size[1]

        self.font = pygame.font.Font(None, 26)

        self.fg_offset = 0
        self.base_shift = 0

        pygame.display.set_caption(name)
    
    def load_assets(self):
        self.load_sprites()
    
    def load_sprites(self):
        self.bg = load_sprite(backgrounds[random.randrange(0,len(backgrounds))])
        self.fg = load_sprite('base')
        #Calculate fg offset
        self.fg_shift = self.fg.get_size()[0] - self.bg.get_size()[0]

    def draw_bg(self):
        self.screen.blit(self.bg, (0,0))

    def draw_fg(self):
        self.screen.blit(self.fg, (self.base_shift,self.height - self.fg.get_size()[1]))
        self.base_shift = -((-self.base_shift + 4) % self.fg_shift)

    def draw_fps(self):
        fps = self.font.render('FPS: ' + str(int(self.clock.get_fps())), True, pygame.Color('white'))
        self.screen.blit(fps, (10, 10))

    def draw(self):
        pygame.display.update()
