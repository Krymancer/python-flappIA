import pygame
import os

cwd = os.getcwd()
mkpath = os.path.join
sprites_path = mkpath(cwd,'assets','sprites')
sprites_extension  = 'png'
sounds_path = mkpath(cwd,'assets','audio')
sounds_extension = 'wav'

def load_sprite(sprite):
    return pygame.image.load(mkpath(sprites_path,f'{sprite}.{sprites_extension}')).convert_alpha()

def load_sound(sound):
    return pygame.mixer.Sound(mkpath(sounds_path,f'{sound}.{sounds_extension}'))