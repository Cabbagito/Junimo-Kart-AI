from torch import nn, optim, from_numpy, save, no_grad
from torch.cuda import is_available
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


IMG_SIZE = (30, 15)  # 30x15 original
PER_CLASS_LIMIT = 200
EPOCHS = 300
BATCH_SIZE = 10000
LEARNING_RATE = 0.001
NUM_MODELS = 10
PLOT = False
TRAIN_TEST_SPLIT = 0.8
device = "cuda" if is_available() else "cpu"
labels = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "None": 10,
}


def bar_plot_split(y):
    plt.bar(
        labels.keys(),
        [len(y[y == labels[label]]) for label in labels.keys()],
    )
    plt.title("Test Distribution")
    plt.show()


def test_split_difference(y):
    counts = [len(y[y == labels[label]]) for label in labels.keys()]
    return max(counts) - min(counts)


def get_data(per_class_limit=1000, test_train_split=TRAIN_TEST_SPLIT, device=device):
    x = []
    y = []
    for directory in os.listdir("DigitClassification/Digits"):
        count = 0
        for file in os.listdir(f"DigitClassification/Digits/{directory}"):
            if count > per_class_limit:
                break
            count += 1
            image = Image.open(f"DigitClassification/Digits/{directory}/{file}")
            image = image.resize(IMG_SIZE)
            image = np.array(image)
            x.append(image)
            y.append(np.array(labels[directory]))

    perm = np.random.permutation(len(x))
    x = np.array(x, dtype=np.float32)[perm].reshape(len(x), -1) / 255
    y = np.array(y, dtype=np.int64)[perm]
    x_train = from_numpy(x[: int(len(x) * test_train_split)]).to(device)
    y_train = from_numpy(y[: int(len(y) * test_train_split)]).to(device)
    x_test = from_numpy(x[int(len(x) * test_train_split) :]).to(device)
    y_test = from_numpy(y[int(len(y) * test_train_split) :]).to(device)

    return x_train, y_train, x_test, y_test


def make_model():
    model = nn.Sequential(
        nn.Linear(IMG_SIZE[0] * IMG_SIZE[1], 256), nn.ReLU(), nn.Linear(256, 11)
    )
    return model


def test_model(model, x, y):
    output = model(x)
    accuracy = (output.argmax(1) == y).float().mean().item()
    return accuracy


def save_model(model, model_number):
    save(model, f"DigitClassification/Models/model{model_number}.model")


def train(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    model_number,
    epochs=2,
    lr=0.001,
    plot_metrics=False,
):

    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for epoch in range(epochs):

        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_fn(output, y_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        with no_grad():
            accuracy = test_model(model, x_test, y_test)
            accuracies.append(accuracy)

    print(f"Model: {model_number}")
    print(f"Loss: {losses[-1]}")
    print(f"Accuracy: {accuracies[-1]}")

    if plot_metrics:
        plt.plot(losses, label="Loss")
        plt.plot(accuracies, label="Accuracy")
        plt.title("Metrics")
        plt.legend()
        plt.show()
    return model


while True:
    x_train, y_train, x_test, y_test = get_data(
        per_class_limit=PER_CLASS_LIMIT, device=device
    )

    bar_plot_split(y_test)
    happy = input("Are you happy with the test split? (y/n) ")
    if happy == "y":
        break

for i in range(NUM_MODELS):
    model = make_model().to(device)
    model = train(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        i + 1,
        epochs=EPOCHS,
        plot_metrics=PLOT,
        lr=LEARNING_RATE,
    )
    save_model(model, i + 1)
