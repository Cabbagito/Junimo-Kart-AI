from time import sleep
from game_input import get_screen_and_score, do_action
from torch import load, Tensor
from agent_model import AgentModel
import matplotlib.pyplot as plt
import numpy as np

start_waiting_time = 2

agent = AgentModel()




digit_classifier = load("DigitClassification/digit_classification.model")
digit_classifier.eval()
digit_classifier.to("cpu")

for i in range(int(start_waiting_time)):
    print(f"Starting in {start_waiting_time - i} seconds...")
    sleep(1)


# while True:

state, score = get_screen_and_score(model=digit_classifier)

state = np.concatenate((state, state, state, state), axis=1)


prediction = agent(Tensor(state))
print(prediction.shape)
