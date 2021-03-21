import numpy as np
import random

from model import QNetwork
from replay_buffer import ReplayBuffer, PrioritizedBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-2              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, mode="train"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            mode (str): if eval, the agent will not learn and collect experiences
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Caches the expected action value of the last act
        self.last_action_value = None
        
        self.mode = self.set_mode(mode)
    
    def step(self, state, action, reward, next_state, done):
        
        if self.mode == "train":
        
            error = self.calculate_error_eval(state, action, reward, next_state, done)

            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done, error)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if self.mode == "train":
            eps = 0

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        self.last_action_value = action_values[0][action].item()
        
        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        with torch.no_grad():
            local_max = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            targets = self.qnetwork_target(next_states).detach().gather(1, local_max)
            target_values = rewards + gamma*targets*(1-dones)

        predicted_values = self.qnetwork_local(states).gather(1, actions)
        
        criterion = torch.nn.MSELoss()
        loss = criterion(predicted_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        self.memory.update_errors((target_values - predicted_values).squeeze())
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def calculate_error_eval(self, state, action, reward, next_state, done):
        """Calculates the error for a given step."""
        self.qnetwork_target.eval()
        
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            target = self.qnetwork_target(next_state).max()

            target_value = reward + GAMMA*target*(1-done)
            error = (target_value - self.last_action_value).item()
        
        self.qnetwork_target.train()
        
        return error

    def save(self, path=""):
        torch.save(self.qnetwork_local.state_dict(), path+"checkpoint_local.pth")
        torch.save(self.qnetwork_target.state_dict(), path+"checkpoint_target.pth")

    def load(self, path=""):
        self.qnetwork_local.load_state_dict(torch.load(path+"checkpoint_local.pth"))
        self.qnetwork_target.load_state_dict(torch.load(path+"checkpoint_target.pth"))
        
    def set_mode(self, mode):
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be one of [train, eval]")
            
        self.mode = mode
    