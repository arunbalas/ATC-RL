# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:17:43 2020

@author: Arun
"""

import gym
from gym import spaces
from gym.utils import seeding
import time
import numpy as np
import pydodo
import json
from sklearn.neighbors import NearestNeighbors
import time

class SimurghEnv(gym.Env):
    """Simple 2 flight environment

    ...

    """

    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.seed()

    def obsr(self, obs):
        obs = np.array(obs)
        state = obs[:,2:]
        return np.array(state.tolist(),dtype=np.float32)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. 

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided to the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        """
  
        pydodo.change_speed('DELTA210', action[0][0])
        pydodo.change_speed('DELTA426', action[1][0])
        time.sleep(0.1)    
        pydodo.simulation_step()

        
        #REWARD:
          
        separations = pydodo.loss_of_separation("DELTA210", "DELTA426")
        r1 = np.abs((action[0][0]-action[1][0])) + separations  
        r2 = np.abs((action[0][0]-action[1][0])) + separations
        reward = [[r1],[r2]]
        #print(reward)
        done = [[(pydodo.aircraft_position("DELTA210")['latitude']<50.9).bool()], [(pydodo.aircraft_position("DELTA426")['longitude']>0.5).bool()]]
        
        # observation, reward, done, info
        return pydodo.all_positions(), reward, done



    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        pydodo.reset_simulation()
        pydodo.upload_sector('sector-X-sector-X-140-400.geojson', 'test_sector')
        pydodo.upload_scenario('cartesian_2agent.json', 'test_scenario')


    def close(self):
        """Override close in the subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pydodo.pause_simulation()
        pydodo.episode_log()
