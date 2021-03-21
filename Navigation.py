# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: drlnd
#     language: python
#     name: drlnd
# ---

# # Navigation
#
# ---
#
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
#
# ### 1. Start the Environment
#
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

from unityagents import UnityEnvironment
import numpy as np
import sys

# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
#
# - **Mac**: `"path/to/Banana.app"`
# - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
# - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
# - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
# - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
# - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
# - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`
#
# For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Banana.app")
# ```

env = UnityEnvironment(file_name="Banana.app")

# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# ### 2. Examine the State and Action Spaces
#
# The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# - `0` - walk forward 
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
#
# The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 
#
# Run the code cell below to print some information about the environment.

# +
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
# -

# ### 4. It's Your Turn!
#
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```

from src.dqn_agent import Agent

agent = Agent(state_size, action_size, 1)


def monitor(env, agent, n_episodes=1000):
    """Run the agent and monitor its performance"""
    scores = []
    best_score = -np.inf
    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:
            action = agent.act(state, 1/(episode+1))

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state

            if done:
                scores.append(score)
                break
                
        if np.mean(scores[-100:]) > best_score:
            best_score = np.mean(scores[-100:])

        print("\rEpisode {} || Best average reward {}".format(episode, best_score), end="")
        sys.stdout.flush()

        if (episode+1) % 50 == 0:
            print("\rEpisode: {}  Mean score: {}                          ".format(episode+1, np.mean(scores[-50:])))

        if np.mean(scores[-100:]) > 13:
            print("\rEnvironment solved in {} episodes!                   ".format(episode+1))
            break
            
    return scores


agent.load("checkpoints/")
agent.set_mode("eval")

score = monitor(env, agent, 1000)

import seaborn as sns

sns.lineplot(x=range(len(score)), y=np.array(score))


