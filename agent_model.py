import torch
from torch.nn import (
    Conv2d,
    MaxPool2d,
    Linear,
    ReLU,
    Sequential,
    Module,
    Flatten,
    Dropout,
    Dropout2d,
)


class AgentModel(Module):
    def __init__(self,  num_actions=9):
        super().__init__()
        self.conv1 = Conv2d(4, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = Linear(256 * 15 * 15, 2048)
        self.fc2 = Linear(2048, 1024)
        self.fc3 = Linear(1024, 256)
        self.fc4 = Linear(256, num_actions)

        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.dropout2d = Dropout2d(p=0.1)

        


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(self.dropout2d(x))))
        x = self.pool(self.relu(self.conv3(self.dropout2d(x))))
        x = self.pool(self.relu(self.conv4(self.dropout2d(x))))
        x = self.flatten(x)

        x = self.relu(self.fc1(self.dropout(x)))
        x = self.relu(self.fc2(self.dropout(x)))
        x = self.relu(self.fc3(self.dropout(x)))
        x = self.fc4(x)
  

    

        return x
