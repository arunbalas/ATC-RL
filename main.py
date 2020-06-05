import gym
import time
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import envs.atc.atc_gym
%matplotlib inline

#Make environment
env = gym.make('AtcEnv-v0')
num_agent = 1

#Take random actions
state = env.reset()                       # reset the environment    
scores = np.zeros(num_agent)              # initialize the score 
while True:
    action = env.action_space.sample()    # select an action 
    next_state, reward, done, info = env.step(action) # send all actions to the environment
    #env.render()
    scores += reward                         # update the score 
    state = next_state
    if np.any(done):              # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

#Call Agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

#Defing DDPG
def ddpg(n_episodes=1500, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()   
        agent.reset()
        score = np.zeros(num_agent)
        for t in range(max_t):
		 # select an action
            action = agent.act(state).reshape(3) 
		 # send all actions to the environment           
            next_state, reward, done, info = env.step(action)           
            #env.render()
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward
            if np.any(done):
                break
                
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        if np.mean(scores_deque) >=300:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
