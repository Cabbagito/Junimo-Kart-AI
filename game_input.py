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


def get_screen_and_score(
    size=(256, 256),
    region=(40, 72, 1235, 710),
    score_region=(100, 5, 200, 35),
    num_digits=5,
):
    img = grab().convert("L").crop(region)
    return img.resize(size), get_score(img.crop(score_region), num_digits=num_digits)


def get_score(score_img, num_digits=5):
    score = ""
    for i in range(num_digits):
        x = 4 + i * 15
        digit_image = score_img.crop((x, 0, x + 15, 30))
        digit = get_digit(digit_image)
        if digit is not None:
            score += digit
        else:
            break
    return int(score)


def get_digit(image):
    from random import randint

    return str(randint(0, 9))


