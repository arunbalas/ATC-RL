# ATC-RL
This project is about application of reinforcement learning to air traffic control. In practice, air traffic control operators (ATCOs) monitor a given sector and issue commands to aircraft pilots to ensure safe separation between aircrafts. They also have to consider the number and frequency of instructions issued, fuel efficiency and orderly handover between sectors. Hence, to learn such complex behavior and automate the systems, Deep Reinforcement Learning is utilized in this project. 

## Note: 
This project will be a part of my paper titled ***"Deep Reinforcement Learning for System Reliability and Intelligence: Applications and Complexities."*** The paper will be published by the end of this year and link to the paper will be updated soon!!

<p align="center">
  <img width="290" height="300" src="https://github.com/arunbalas/ATC-RL/blob/develop/Images/Final%20single.gif">
  <img width="290" height="300" src="https://github.com/arunbalas/ATC-RL/blob/develop/Images/Final%20Gif1.gif">
  <img width="290" height="300" src="https://github.com/arunbalas/ATC-RL/blob/develop/Images/Final%20Gif.gif">
</p>


## Requisites & Installation

## Clone my repository:

```bash
git clone --branch develop https://github.com/arunbalas/ATC-RL.git
```


### Dependencies

### 1. Install python packages

```bash
pip install -r requirements.txt
```

### 2. Install Docker
Make sure you have [Docker](https://www.docker.com/get-started) installed.



### 3. Install PyDodo

Pydodo is required for multi-agent systems since the environment depends on bluesky simulator.

To install:

```bash
cd ATC-RL/Multi-Agent
git clone https://github.com/alan-turing-institute/dodo.git
pip install dodo/PyDodo
```

## Train Agent
The python file main.py (located in respective Single-Agent and Multi-Agent folders) will initiate the training. To see the visualization, please use Twitcher (http://localhost:8080/). 

### For Single-Agent training:
```bash
cd ATC-RL/Single Agent
python main.py
```
### For Multi-Agent training:
```bash
cd ATC-RL/Single Agent
python main.py
```

### Sample multi-agent performance with random starting locations:
The RL algorithm is tested for 1000 different random locations to check the model performance for unseen scenarios. The number of crashes were around 90, but the performance was not that bad given the training time. This can certainly be improved by modifying the parameters and adding some more action variables. The verification.py file will run the trained agent with different starting locations. The user can change the number of episodes to run.

<!--- ![sample_gif](https://github.com/arunbalas/ATC-RL/blob/develop/Final%20Gif.gif) -->

<p align="center">
  <img width="460" height="300" src="https://github.com/arunbalas/ATC-RL/blob/develop/Images/Final%20Gif.gif">
</p>


#### References:
[![DOI](https://zenodo.org/badge/148370950.svg)](https://zenodo.org/badge/latestdoi/148370950)
[![DOI](https://travis-ci.com/alan-turing-institute/simurgh.svg?branch=master)](https://travis-ci.com/alan-turing-institute/simurgh)
- [Bluebird](https://github.com/alan-turing-institute/bluebird)
- [Twitcher](https://github.com/alan-turing-institutetwitcher)
- [Dodo](https://github.com/alan-turing-institute/dodo)
- [Aviary](https://github.com/alan-turing-institute/aviary)

