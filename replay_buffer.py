import numpy as np
import pandas as pd
import random
import torch

from collections import namedtuple, deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
class PrioritizedBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed, a=0.01, e=1):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error"])
        self.seed = random.seed(seed)
        self.a = a
        self.e = e
        self.experience_indices = None
    
    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, error)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        self.experience_indices = np.random.choice(range(len(self.memory)), size=self.batch_size, replace=False, p=self.get_weights())
        
        experiences = [item for index, item in enumerate(self.memory) if index in self.experience_indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
        
    def get_weights(self):
        weights = (np.abs(np.array([e.error for e in self.memory if e is not None])) + self.a) ** self.e
        return weights / weights.sum()
    
    def update_errors(self, errors):
        for index, error in zip(self.experience_indices, errors):
            state, action, reward, next_state, done, _ = self.memory[index]
            self.memory[index] = self.experience(state, action, reward, next_state, done, error.item())

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        