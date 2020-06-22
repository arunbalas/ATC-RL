# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:30:36 2020

@author: Arun
"""


#import Simurgh-multi-agent-main
from Simurgh_multi_agent_main import mddpg


################################
##                            ##
##          Arun B S          ##
##    github.com/arunbalas    ##
##                            ##
################################


if __name__ == "__main__":
    scores = mddpg(n_episodes=1500, max_t=1000, print_every=10)