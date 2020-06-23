# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:17:43 2020

@author: Arun
"""



#%%
import os
os.system('docker-compose --file ../docker-compose.yml up --detach')
import pydodo
import time
import torch
import numpy as np
from maddpg_agent import Agent
from env import SimurghEnv
import streamlit as st
import pydodo
import streamlit as st
import pandas as pd
st.write(pydodo.bluebird_connect.get_bluebird_url())
print(pydodo.bluebird_connect.get_bluebird_url())
help(pydodo.bluebird_config)
st.write(pydodo.all_positions())
pydodo.simulation_info()
pydodo.upload_sector('sector-X-sector-X-140-400.geojson', 'test_sector')
pydodo.upload_scenario('cartesian_2agent.json', 'test_scenario')


#%%
#Initialize
state_size = 8
action_size = 1
env = SimurghEnv()
env.observation_space = state_size
env.action_space = action_size
num_agents = 2
agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location='cpu'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location='cpu'))

#get environment
env = SimurghEnv()
obs = pydodo.all_positions()                                     
progress_bar = st.progress(0)
status_text = st.empty()
pydodo.simulation_step()

#create map
map_data = pd.DataFrame([np.array(obs['latitude']), np.array(obs['longitude'])]).transpose()
map_data.columns=['lat','lon']
chart = st.map(map_data, zoom = 5) 
pydodo.set_simulation_rate_multiplier(20)

for j in range(10):
    j=0
    env.reset()
    state = np.array(env.obsr(obs))
    score = np.zeros(num_agents)                          # initialize the score 
    while True:
        pydodo.set_simulation_rate_multiplier(20)
        progress_bar.progress(j)        
        action =(agent.act(state)).astype(int) 
        print(action) 
        nxt_state, reward, done = env.step(action.tolist()) 
        next_state = env.obsr(nxt_state)
        agent.step(state, action, reward, next_state, done)
        state = np.round(next_state,2)
        print(pydodo.all_positions())
        
        obs = pydodo.all_positions()
        new_rows = pd.DataFrame([np.array(obs['latitude']), np.array(obs['longitude'])]).transpose()
        new_rows.columns=['lat','lon']
        # Update status text.
        status_text.text(new_rows)
    
        # Append data to the chart.
        chart.map(new_rows)
        j= j+1
        score += np.sum(reward)
        if np.any(done):
            break
        # while True
        #     return obs
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))




status_text.text('Done!')
st.balloons()