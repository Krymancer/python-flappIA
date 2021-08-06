import sys
import pygame as pg
import random
from itertools import cycle


GAMENAME = "Flappy Bird"
FPS = 60
SCREENHEIGHT = 512
SCREENWIDTH = 288

PIPEGAP = 100
BASEY = SCREENHEIGHT * 0.79

IMAGES,SOUNDS,HITMASKS = {}, {}, {}

BASE_SPRITE = 'assets/sprites'
BASE_AUDIO = 'assets/audio'

BIRDS_LIST = (
    (
        f'{BASE_SPRITE}/redbird-upflap.png',
        f'{BASE_SPRITE}/redbird-midflap.png',
        f'{BASE_SPRITE}/redbird-downflap.png'
    ),
    (
        f'{BASE_SPRITE}/bluebird-upflap.png',
        f'{BASE_SPRITE}/bluebird-midflap.png',
        f'{BASE_SPRITE}/bluebird-downflap.png'
    ),
    (
        f'{BASE_SPRITE}/yellowbird-upflap.png',
        f'{BASE_SPRITE}/yellowbird-midflap.png',
        f'{BASE_SPRITE}/yellowbird-downflap.png'
    )
)

BACKGROUND_LIST = (
    f'{BASE_SPRITE}/background-day.png',
    f'{BASE_SPRITE}/background-night.png'
)

PIPES_LIST = (
    f'{BASE_SPRITE}/pipe-green.png',
    f'{BASE_SPRITE}/pipe-red.png'
)

ICON = f'{BASE_SPRITE}/flappy.ico'

def main():
    global SCREEN,CLOCK
    pg.init()
    SCREEN = pg.display.set_mode((SCREENWIDTH,SCREENHEIGHT))
    CLOCK = pg.time.Clock()
    pg.display.set_caption(GAMENAME)
    icon = pg.image.load(ICON)
    pg.display.set_icon(icon)

    IMAGES['numbers'] = (
        pg.image.load(f'{BASE_SPRITE}/0.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/1.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/2.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/3.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/4.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/5.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/6.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/7.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/8.png').convert_alpha(),
        pg.image.load(f'{BASE_SPRITE}/9.png').convert_alpha()
    )

    IMAGES['gameover'] = pg.image.load(f'{BASE_SPRITE}/gameover.png')
    IMAGES['message'] = pg.image.load(f'{BASE_SPRITE}/message.png')
    IMAGES['base'] = pg.image.load(f'{BASE_SPRITE}/base.png')

    SOUNDS['die'] = pg.mixer.Sound(f'{BASE_AUDIO}/die.ogg')
    SOUNDS['hit'] = pg.mixer.Sound(f'{BASE_AUDIO}/hit.ogg')
    SOUNDS['point'] = pg.mixer.Sound(f'{BASE_AUDIO}/point.ogg')
    SOUNDS['swoosh'] = pg.mixer.Sound(f'{BASE_AUDIO}/swoosh.ogg')
    SOUNDS['wing'] = pg.mixer.Sound(f'{BASE_AUDIO}/wing.ogg')

    while True:
        random_background = random.randint(0,len(BACKGROUND_LIST)-1)
        IMAGES['background'] = pg.image.load(BACKGROUND_LIST[random_background]).convert()

        random_bird = random.randint(0,len(BIRDS_LIST)-1)
        IMAGES['bird'] = (
            pg.image.load(BIRDS_LIST[random_bird][0]).convert_alpha(),
            pg.image.load(BIRDS_LIST[random_bird][1]).convert_alpha(),
            pg.image.load(BIRDS_LIST[random_bird][2]).convert_alpha()
        )

        random_pipe = random.randint(0,len(PIPES_LIST)-1)
        IMAGES['pipes'] = (
            pg.transform.flip(pg.image.load(PIPES_LIST[random_pipe]).convert_alpha(),False,True),
            pg.image.load(PIPES_LIST[random_pipe]).convert_alpha()
        )
        
        HITMASKS['pipe'] = (
            get_hitmasks(IMAGES['pipes'][0]),
            get_hitmasks(IMAGES['pipes'][1])
        )

        HITMASKS['bird'] = (
            get_hitmasks(IMAGES['bird'][0]),
            get_hitmasks(IMAGES['bird'][1]),
            get_hitmasks(IMAGES['bird'][2])
        )

        moviment_info = show_welcome_animation()
        crash_info = main_game(moviment_info)
        show_gameover_screen(crash_info)


def show_welcome_animation():
    """Shows welcome screen animation of flappy bird"""
    current_bird_sprite = 0
    bird_sprite_animation_generator = cycle([0,1,2,1])
    animation_iterator = 0

    bird_x = SCREENWIDTH * 0.2
    bird_y = (SCREENHEIGHT - IMAGES['bird'][0].get_height()) / 2

    message_x = (SCREENWIDTH - IMAGES['message'].get_width()) / 2
    message_y = (SCREENHEIGHT * 0.12)

    base_x = 0
    base_shift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    bird_shm_vals = {'value': 0, 'direction': 1}

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN and (event.key == pg.K_SPACE or event.key == pg.K_UP):
                # make first flap sound and return values for mainGame
                SOUNDS['wing'].play()

                return {
                    'bird_y': bird_y + bird_shm_vals['value'],
                    'base_x': base_x,
                    'bird_sprite_animation_generator': bird_sprite_animation_generator,
                }

        if (animation_iterator + 1) % 5 == 0:
            current_bird_sprite = next(bird_sprite_animation_generator)
        animation_iterator = (animation_iterator + 1) % 30
        base_x = -((-base_x + 4) % base_shift)
        bird_smh(bird_shm_vals)

        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['bird'][current_bird_sprite],
                    (bird_x, bird_y + bird_shm_vals['value']))
        SCREEN.blit(IMAGES['message'], (message_x, message_y))
        SCREEN.blit(IMAGES['base'], (base_x, BASEY))

        pg.display.update()
        CLOCK.tick(FPS)

def main_game(moviment_info):
    score = current_bird_sprite = animation_iterator = 0
    bird_sprite_animation_generator = moviment_info['bird_sprite_animation_generator']
    bird_x, bird_y = (SCREENWIDTH * 0.2), moviment_info['bird_y']

    base_x = moviment_info['base_x']
    base_shift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    new_pipe1 = get_random_pipe()
    new_pipe2 = get_random_pipe()

    upper_pipes = [
        {'x': SCREENWIDTH + 200, 'y': new_pipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH/2), 'y': new_pipe2[0]['y']},
    ]

    lower_pipes = [
        {'x': SCREENWIDTH + 200, 'y': new_pipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': new_pipe2[1]['y']},
    ]

    pipe_vel_x = -4

    bird_vel_y = -9
    bird_max_vel_y = 10
    bird_min_vel_y = -8
    bird_acceleration_y = 1
    bird_rotation = 45
    bird_rotation_velocity = 3
    bird_rotation_threshold = 20
    bird_flap_acceleration = -9
    bird_flapped = False 

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN and (event.key == pg.K_SPACE or event.key == pg.K_UP):
                if bird_y > -2 * IMAGES['bird'][0].get_height():
                    bird_vel_y = bird_flap_acceleration
                    bird_flapped = True
                    SOUNDS['wing'].play()

        crash_test = check_crash({'x':bird_x,'y':bird_y,'index':current_bird_sprite},upper_pipes,lower_pipes)

        if crash_test[0]:
            return {
                'y': bird_y,
                'ground_crash': crash_test[1],
                'base_x': base_x,
                'upper_pipes': upper_pipes,
                'lower_pipes': lower_pipes,
                'score': score,
                'bird_vel_y': bird_vel_y,
                'bird_rotation': bird_rotation
            }

        bird_mid_pos = bird_x + IMAGES['bird'][0].get_width() / 2
        for pipe in upper_pipes:
            pipe_mid_pos = pipe['x'] + IMAGES['pipes'][0].get_width() / 2
            if pipe_mid_pos <= bird_mid_pos < pipe_mid_pos + 4:
                score += 1
                SOUNDS['point'].play()

        if (animation_iterator + 1) % 3 == 0:
            current_bird_sprite = next(bird_sprite_animation_generator)

        animation_iterator = (animation_iterator + 1) % 30
        base_x = -((-base_x + 100) % base_shift) 

        if bird_rotation > -90:
            bird_rotation -= bird_rotation_velocity

        if bird_vel_y < bird_max_vel_y and not bird_flapped:
            bird_vel_y += bird_acceleration_y
        if bird_flapped:
            bird_flapped = False

            bird_rotation = 45

        bird_height = IMAGES['bird'][current_bird_sprite].get_height()
        bird_y += min(bird_vel_y,BASEY - bird_y -bird_height)

        #move pipes
        for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
            up_pipe['x'] += pipe_vel_x
            low_pipe['x'] += pipe_vel_x

        if len(upper_pipes) > 0 and 0 < upper_pipes[0]['x'] < 5:
            new_pipe = get_random_pipe()
            upper_pipes.append(new_pipe[0])
            lower_pipes.append(new_pipe[1])

        if len(upper_pipes) > 0 and upper_pipes[0]['x'] < -IMAGES['pipes'][0].get_width():
            upper_pipes.pop(0)
            lower_pipes.pop(0)

        # draw
        SCREEN.blit(IMAGES['background'], (0,0))

        for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
            SCREEN.blit(IMAGES['pipes'][0], (up_pipe['x'], up_pipe['y']))
            SCREEN.blit(IMAGES['pipes'][1], (low_pipe['x'], low_pipe['y']))

        SCREEN.blit(IMAGES['base'], (base_x, BASEY))
        
        show_score(score)

        visible_bird_rotation = bird_rotation_threshold

        if bird_rotation <= bird_rotation_threshold:
            visible_bird_rotation = bird_rotation

        bird_surface = pg.transform.rotate(IMAGES['bird'][current_bird_sprite], visible_bird_rotation)
        SCREEN.blit(bird_surface,(bird_x,bird_y))

        pg.display.update()
        CLOCK.tick(FPS)

    pass

def show_gameover_screen(crash_info):
    """crashes the player down ans shows gameover image"""
    score = crash_info['score']
    bird_x = SCREENWIDTH * 0.2
    bird_y = crash_info['y']
    bird_height = IMAGES['bird'][0].get_height()
    bird_vel_y = crash_info['bird_vel_y']
    bird_acceleration = 2
    bird_rotation = crash_info['bird_rotation']
    bird_rotation_velocity = 7

    base_x = crash_info['base_x']

    upper_pipes, lower_pipes = crash_info['upper_pipes'], crash_info['lower_pipes']

    # play hit and die sounds
    SOUNDS['hit'].play()
    if not crash_info['ground_crash']:
        SOUNDS['die'].play()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN and (event.key == pg.K_SPACE or event.key == pg.K_UP):
                if bird_y + bird_height >= BASEY - 1:
                    return

        if bird_y + bird_height < BASEY - 1:
            bird_y += min(bird_vel_y, BASEY - bird_y - bird_height)

        if bird_vel_y < 15:
            bird_vel_y += bird_acceleration

        if not crash_info['ground_crash']:
            if bird_rotation > -90:
                bird_rotation -= bird_rotation_velocity

        SCREEN.blit(IMAGES['background'], (0,0))
        for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
            SCREEN.blit(IMAGES['pipes'][0], (up_pipe['x'], up_pipe['y']))
            SCREEN.blit(IMAGES['pipes'][1], (low_pipe['x'], low_pipe['y']))
        SCREEN.blit(IMAGES['base'], (base_x, BASEY))
        show_score(score)

        bird_surface = pg.transform.rotate(IMAGES['bird'][1], bird_rotation)
        SCREEN.blit(bird_surface,(bird_x,bird_y))

        SCREEN.blit(IMAGES['gameover'], (50, 180))

        CLOCK.tick(FPS)
        pg.display.update()

def show_score(score):
    """displays score in center of screen"""
    score_digits = [int(x) for x in list(str(score))]
    total_width = 0 # total width of all numbers to be printed

    for digit in score_digits:
        total_width += IMAGES['numbers'][digit].get_width()

    x_offset = (SCREENWIDTH - total_width) / 2

    for digit in score_digits:
        SCREEN.blit(IMAGES['numbers'][digit], (x_offset, SCREENHEIGHT * 0.1))
        x_offset += IMAGES['numbers'][digit].get_width()


def check_crash(bird,up_pipes,low_pipes):
    """returns True if player collides with base or pipes."""
    bird_index = bird['index']
    bird['w'] = IMAGES['bird'][0].get_width()
    bird['h'] = IMAGES['bird'][0].get_height()

    # if player crashes into ground
    if bird['y'] + bird['h'] >= BASEY - 1:
        return [True, True]
    else:
        bird_rect = pg.Rect(bird['x'], bird['y'],
                      bird['w'], bird['h'])
        pipe_w = IMAGES['pipes'][0].get_width()
        pipe_h = IMAGES['pipes'][0].get_height()

        for up_pipe, low_pipe in zip(up_pipes, low_pipes):
            # upper and lower pipe rects
            up_pipe_rect = pg.Rect(up_pipe['x'], up_pipe['y'], pipe_w, pipe_h)
            low_pipe_rect = pg.Rect(low_pipe['x'], low_pipe['y'], pipe_w, pipe_h)

            # player and upper/lower pipe hitmasks
            bird_hitmask = HITMASKS['bird'][bird_index]
            up_pipe_hitmask = HITMASKS['pipe'][0]
            low_pipe_hitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            up_collide = pixel_collision(bird_rect, up_pipe_rect, bird_hitmask, up_pipe_hitmask)
            low_collide = pixel_collision(bird_rect, low_pipe_rect, bird_hitmask, low_pipe_hitmask)

            if up_collide or low_collide:
                return [True, False]
    return [False, False]

def pixel_collision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def get_random_pipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gap_y = random.randrange(0, int(BASEY * 0.6 - PIPEGAP))
    gap_y += int(BASEY * 0.2)
    pipe_height = IMAGES['pipes'][0].get_height()
    pipe_x = SCREENWIDTH + 10

    return [
        {'x': pipe_x, 'y': gap_y - pipe_height},  # upper pipe
        {'x': pipe_x, 'y': gap_y + PIPEGAP}, # lower pipe
    ]

def bird_smh(bird_smh):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(bird_smh['value']) == 8:
        bird_smh['direction'] *= -1

    if bird_smh['direction'] == 1:
         bird_smh['value'] += 1
    else:
        bird_smh['value'] -= 1


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
            playerShm['val'] += 1
    else:
        playerShm['val'] -= 1

def get_hitmasks(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == "__main__":
    main()