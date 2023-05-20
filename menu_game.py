# C:\Users\Admin\AppData\Local\Programs\Python\Python39\python.exe .\menu_game.py

__all__ = ['main']

import pygame
import pygame_menu
from pygame_menu.examples import create_example_window

from typing import Any
from functools import partial
from start_game import *


FPS = 30
WINDOW_SIZE = (800, 600)
runmenu_game = False


# varglobal.menu.close()

class VAR_GLOBAL:
    menu = None
    def __init__(self) -> None:
        self.name = 'VAR_G'  
        self.choose_round = 1


varglobal = VAR_GLOBAL()
st_game = START_GAME()

def on_button_click(value: str, text: Any = None) -> None:
    """
    Button event on menus.
    :param value: Button value
    :param text: Button text
    """
    if not text:
        if value == "START_CAMERA":
            st_game.run = True 
            st_game.run_start_game_choose_camera(varglobal.choose_round)
            print(value)
        elif "CAUHINH_" in value:
            value = str.split(value,'_')
            varglobal.choose_round = int(value[1])
            print('Choose ' + value[1] + ' round')
        elif value == "START_HINHANH":
            st_game.run = True
            st_game.run_start_game_choose_hinhanh(varglobal.choose_round)
            print(value)
            
    else:
        print('else')
        


def paint_background(surface: 'pygame.Surface') -> None:
    """
    Paints a given surface with background color.
    :param surface: Pygame surface
    """
    surface.fill((128, 230, 200))


def make_long_menu() -> 'pygame_menu.Menu':
    """
    Create a long scrolling menu.
    :return: Menu
    """
    theme_menu = pygame_menu.themes.THEME_BLUE.copy()
    theme_menu.scrollbar_cursor = pygame_menu.locals.CURSOR_HAND

    # Main menu, pauses execution of the application
    menu = pygame_menu.Menu(
        height=400,
        onclose=pygame_menu.events.EXIT,
        theme=theme_menu,
        title='MÀN HÌNH CHÍNH',
        width=600
    )

    start_button = pygame_menu.Menu(
        columns=4,
        height=400,
        onclose=pygame_menu.events.EXIT,
        rows=4,
        theme=pygame_menu.themes.THEME_GREEN,
        title='Bắt đầu',
        width=600
    )

    


    cauhinh_button = pygame_menu.Menu(
        height=400,
        onclose=pygame_menu.events.EXIT,
        theme=pygame_menu.themes.THEME_DARK,
        title='Cấu hình',
        width=600
    )

    menu.add.button('Bắt đầu', start_button)
    menu.add.button('Cấu hình', cauhinh_button)
    menu.add.button('Thoát', pygame_menu.events.EXIT)
    menu.add.vertical_margin(20)  # Adds margin


    start_button.add.button('Chế độ dự đoán bằng camera',on_button_click,'START_CAMERA')
    start_button.add.button('Chế độ chọn hình ảnh',on_button_click,'START_HINHANH')
    

    start_button.add.button('Back', pygame_menu.events.BACK)

    
    cauhinh_button.add.label(
        '         Hãy chọn số lượt chơi với máy',
        max_char=40,
        align=pygame_menu.locals.ALIGN_LEFT,
        margin=(0, -1)
    )
    for i in range(1, 10):
        str_button_cauhinh = "CAUHINH_" + str(i)
        cauhinh_button.add.button(str(i),on_button_click,str_button_cauhinh)

    cauhinh_button.add.button('Back', pygame_menu.events.BACK)

    return menu


def main(test: bool = False) -> None:
    
    screen = create_example_window('GAME - nguyễn Hải Long', WINDOW_SIZE)

    clock = pygame.time.Clock()
    menu_mainfunc = make_long_menu()
    varglobal.menu = menu_mainfunc
    
    while True:
        
        menu_mainfunc.mainloop(
            surface=screen,
            bgfun=partial(paint_background, screen),
            disable_loop=test,
            fps_limit=FPS
        )
        
        pygame.display.flip()
        
        if test:
            break
            


if __name__ == '__main__':
    main(test = False)
    