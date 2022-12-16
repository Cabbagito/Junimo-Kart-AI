import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# def classify_digit(image):
#    pass


def get_data(limit=1000, shuffle=True, to_numpy=False):
    x = []
    y = []
    for directory in os.listdir("DigitClassification/Digits"):
        count = 0
        for file in os.listdir(f"DigitClassification/Digits/{directory}"):
            if count >= limit:
                break
            if to_numpy:
                image = np.array(
                    Image.open(f"DigitClassification/Digits/{directory}/{file}")
                )
            else:
                image = Image.open(f"DigitClassification/Digits/{directory}/{file}")
            x.append(image)
            y.append(directory)
            count += 1

    if shuffle:
        from random import shuffle

        c = list(zip(x, y))
        shuffle(c)
        x, y = zip(*c)

    return x, y


def get_labels(to_numpy=False):
    labels = {}
    for label in os.listdir("DigitClassification/Labels"):
        if to_numpy:
            labels[label[:-4]] = np.array(
                Image.open(f"DigitClassification/Labels/{label}")
            )
        else:
            labels[label[:-4]] = Image.open(f"DigitClassification/Labels/{label}")
    return labels


def bar_plot_data():
    data = {}

    for directory in os.listdir("DigitClassification/Digits"):
        if directory.endswith(".png"):
            continue
        
        data[directory] = os.listdir(
            os.path.join("DigitClassification/Digits", directory)
        ).__len__()

    plt.bar(data.keys(), data.values())
    plt.show()


# x, y = get_data(limit=20, to_numpy=True)

# labels = get_labels(to_numpy=True)

bar_plot_data()
