from time import sleep
from PIL.ImageGrab import grab
from string import ascii_lowercase
from random import choice

region = (40, 72, 1235, 710)
score_region = (100, 5, 200, 35)
num_digits = 5
wait_time = 1
start_waiting_time = 5

def hash():
    return "".join(choice(ascii_lowercase) for _ in range(10))


for i in range(int(start_waiting_time)):
        print(f"Starting in {start_waiting_time - i} seconds...")
        sleep(1)

while True:
    sleep(wait_time)
    screen = grab().convert("L").crop(region)
    score = screen.crop(score_region)
    for i in range(num_digits):
        x = 4 + i * 15
        digit_image = score.crop((x, 0, x + 15, 30))
        digit_image.save(f"DigitClassification/Digits/{hash()}.png")
