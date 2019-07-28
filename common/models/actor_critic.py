import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x 


class ActorConv(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorConv, self).__init__()

        self.c1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.f1 = nn.Linear(64 * 7 * 7, 512)
        self.f2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)    # NHWC -> NCHW
        x = x.float() / 255
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.f1(x.view(x.size(0), -1)))
        return self.f2(x)

class CriticConv(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticConv, self).__init__()
        
        self.c1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.f1 = nn.Linear(64 * 7 * 7, 512)
        self.f2 = nn.Linear(512, 100)
        self.f3 = nn.Linear(100 + action_dim, 50)
        self.f4 = nn.Linear(50, 1)

    def forward(self, x, u):
        x = x.permute(0, 3, 1, 2)    # NHWC -> NCHW
        x = x.float() / 255
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.f1(x.view(x.size(0), -1)))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(torch.cat([x, u], 1)))
        x = self.f4(x)
        return x 
