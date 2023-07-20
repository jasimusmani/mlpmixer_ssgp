import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(25*7*3, 1728)
        self.fc2 = nn.Linear(1728, 864)
        self.fc3 = nn.Linear(864, 432)
        self.fc4 = nn.Linear(432, 128)
        self.fc5 = nn.Linear(128, 25*3)

    def forward(self, x):
        x = self.flatten(x)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x.view(-1, 25, 3)