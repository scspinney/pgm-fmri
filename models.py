import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self,fc_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(np.int(fc_dim), np.int(fc_dim))
        self.fc2 = nn.Linear(np.int(fc_dim), 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim = 0)

        return x
