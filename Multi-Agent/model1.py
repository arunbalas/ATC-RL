import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator



class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=1024, fc2_units=1024):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.blinear1 = BayesianLinear(state_size, fc1_units)
        self.blinear2 = BayesianLinear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.blinear1.weight_mu.data.uniform_(3e-3, 3e-2)
        self.blinear2.weight_mu.data.uniform_(3e-3, 3e-2)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = self.bn1(x_)
        return self.blinear2(x_)



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=1024, fc2_units=1024):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = BayesianLinear(state_size, fcs1_units)
        self.fc2 = BayesianLinear(fcs1_units+action_size, fc2_units)
        self.fc3 = BayesianLinear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight_mu.data.uniform_(3e-3, 3e-2)
        self.fc2.weight_mu.data.uniform_(3e-3, 3e-2)
        self.fc3.weight_mu.data.uniform_(3e-3, 3e-2)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
