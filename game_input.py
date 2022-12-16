from PIL.ImageGrab import grab
from time import sleep
import keyboard
import numpy as np
from torch import Tensor

labels = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: None,
}
moves = [None, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def press(duration=0.01):
    keyboard.press("space")
    sleep(duration)
    keyboard.release("space")


def do_action(action):
    if action == 0:
        return
    keyboard.press("space")
    sleep(moves[action])
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
    model=None,
):
    img = grab().convert("L").crop(region)
    return np.expand_dims(
        np.array(img.resize(size), dtype=np.float32) / 255, axis=[0, 1]
    ), get_score(img.crop(score_region), num_digits=num_digits, model=model)


def get_score(score_img, num_digits=5, model=None):
    score = ""
    for i in range(num_digits):
        x = 4 + i * 15
        digit_image = score_img.crop((x, 0, x + 15, 30))
        digit = get_digit(digit_image, model)
        if digit is not None:
            score += digit
        else:
            break
    return 0 if score == "" else int(score)


def get_digit(
    image,
    model,
):
    image = np.array(image, dtype=np.float32)
    image = image.reshape(1, -1)
    prediction = model(Tensor(image))
    return labels[prediction.argmax().item()]
