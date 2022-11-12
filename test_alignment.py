from time import sleep
from PIL.ImageGrab import grab

def test_alignment(
    wait_for=5,
    size=(256, 256),
    region=(40, 72, 1235, 710),
    score_region=(100, 5, 200, 35),
    num_digits=5,
):
    for i in range(wait_for):
        print(f"Starting in {wait_for - i} seconds...")
        sleep(1)
    screen = grab().convert("L").crop(region)
    screen_resized = screen.resize(size)
    screen_resized.save("TestAlignment/screen.png")
    score = screen.crop(score_region)
    for i in range(num_digits):
        x = 4 + i * 15
        digit_image = score.crop((x, 0, x + 15, 30))
        digit_image.save("TestAlignment/digit{}.png".format(i + 1))


test_alignment()