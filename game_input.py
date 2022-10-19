from PIL.ImageGrab import grab
from time import sleep
import keyboard
import numpy as np


def press(duration=0.01):
    keyboard.press("space")
    sleep(duration)
    keyboard.release("space")


def get_screen(size=(256, 256), region=(40, 72, 1235, 710)):
    return np.array(get_screen_image(size, region))


def get_screen_image(size=(256, 256), region=(40, 72, 1235, 710)):
    return grab().convert("L").crop(region).resize(size)
