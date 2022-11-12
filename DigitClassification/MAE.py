import numpy as np

def classify_digit(image,labels):

    scores = {}

    for label, label_img in labels.items():
        score = np.sum(np.abs(np.subtract(image/255,label_img/255)))
        scores[label] = score

    return min(scores, key=scores.get)

    