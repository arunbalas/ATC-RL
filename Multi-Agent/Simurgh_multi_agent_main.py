#!/usr/bin/env python
# coding: utf-8
"""
Created on Mon Jun 15 19:17:43 2020

@author: Arun
"""

import os
os.system('docker-compose --file docker-compose.yml up --detach')
import pydodo
import gym
import time
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from maddpg_agent import Agent
from gym import spaces
from gym.utils import seeding
import itertools
from pydodo.episode_log import episode_log
from pydodo import change_altitude
from pydodo.metrics import loss_of_separation
from pydodo.request_position import all_positions
from pydodo.simulation_control import simulation_step, reset_simulation, pause_simulation
from sklearn.neighbors import NearestNeighbors
import time
from env import SimurghEnv
import streamlit as st

print(pydodo.bluebird_connect.get_bluebird_url())
print(pydodo.bluebird_connect.get_bluebird_url())
#help(pydodo.bluebird_config)
st.write(pydodo.all_positions())
pydodo.simulation_info()
pydodo.upload_sector('sector-X-sector-X-140-400.geojson', 'test_sector')
pydodo.upload_scenario('cartesian_2agent.json', 'test_scenario')



# In[276]:


env = SimurghEnv()
env.observation_space = 7
env.action_space = 1
num_agents = 2
agent = Agent(state_size=7, action_size=1, random_seed=10)


# In[279]:


def mddpg(n_episodes=1500, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    #i=1
    for i_episode in range(1, n_episodes+1):
        #Take random actions
        env.reset()
        pydodo.set_simulation_rate_multiplier(15)
        obs = pydodo.all_positions()                                 # reset the environment  
        state = np.array(env.obsr(obs))
        #action = [[60],[65]]
        agent.reset()
        score = np.zeros(num_agents)
        while True:
            action =(agent.act(state)).astype(int)            # select action
            print(action)
            st.write(action) 
            nxt_state, reward, done = env.step(action.tolist())           # send all actions to the environment
            next_state = env.obsr(nxt_state)
            agent.step(state, action, reward, next_state, done)
            state = np.round(next_state,2)
            score += np.sum(reward)
            if np.any(done):
                break
                
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            
        if np.mean(scores_deque) >=30000:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores


# if __name__ == "__main__":
#     scores = mddpg()




