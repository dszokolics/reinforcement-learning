import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        n = 128
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.lin_1 = nn.Linear(state_size, n)
        self.lin_2 = nn.Linear(n, n)
        self.lin_3 = nn.Linear(n, n)

        self.lin_adv = nn.Linear(n, action_size)
        self.lin_state = nn.Linear(n, 1)
        
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = F.relu(self.lin_1(state))
        state = self.dropout(state)
        state = F.relu(self.lin_2(state))
        state = F.relu(self.lin_3(state))

        # Dueling architecture
        adv = self.lin_adv(state)
        st = self.lin_state(state)
        
        adv = adv - adv.mean(1)[0]
        
        return adv+st
