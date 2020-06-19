# ATC-RL
This project is about application of reinforcement learning to real time system control. In practice, air traffic control operators (ATCOs) monitor a given sector and issue commands to aircraft pilots to ensure safe separation between aircrafts. They also have to consider the number and frequency of instructions issued, fuel efficiency and orderly handover between sectors. Hence, to learn such complex behavior and solve such environments, Deep Reinforcement Learning is utilized in this project.


## Requisites & Installation

Please refer to requirements.txt file.

```bash
pip install -r requirements.txt
```


### Dependencies

- Need Bluesky, Bluebird and Pydodo which can be easily installed by following the steps given below:

### 1. Clone Simurgh repository


```{bash}
git clone https://github.com/alan-turing-institute/simurgh.git
```

All commands described in the subsequent sections are meant to be run from inside the repo. After cloning the repo make sure to run:

```{bash}
cd simurgh
```

### 2. Run BlueBird, BlueSky & Twicher with Docker

Make sure you have [Docker](https://www.docker.com/get-started) installed.

If you have Docker installed and have cloned this repo then run:

```{bash}
docker-compose up -d
```

This pulls down the pre-built images from DockerHub and
starts each container in the right order.

Then all one needs to do is go to
`http://localhost:8080` where Twitcher will be running.

_Note_: If this is the first time running this command, it may take some time to
download and extract all the layers involved.

Then to close this, run:

```
docker-compose down
```

This will shutdown the running instances.

### 3. Install PyDodo

PyDodo is the Python implementation of Dodo.

To install:

```bash
git clone https://github.com/alan-turing-institute/dodo.git
pip install dodo/Pydodo
```

## Clone my repository (Note: This repository needs to be clone into simurgh folder):
```bash
git clone --branch develop https://github.com/arunbalas/ATC-RL.git
```

## Train Agent
The python file main.py will initiate the training. To see the training and validation use Twitcher (http://localhost:8080/). It is also possible to see the validation running in streamlit (http://localhost:8501/).

```bash
python main.py
```
### Sample performance with random starting locations:
The RL algorithm is tested for 1000 different random locations to check the model performance for unseen scenarios. The number of crashes were around 90, but the performance was not that bad given the training time. This can certainly be improved by modifying the parameters and adding some more action variables. The verification.py file will run the trained agent with different starting locations. The user can change the number of episodes to run.

<!--- ![sample_gif](https://github.com/arunbalas/ATC-RL/blob/develop/Final%20Gif.gif) -->

<p align="center">
  <img width="460" height="300" src="https://github.com/arunbalas/ATC-RL/blob/develop/Final%20Gif.gif">
</p>



#### References:
[![DOI](https://zenodo.org/badge/148370950.svg)](https://zenodo.org/badge/latestdoi/148370950)
[![DOI](https://travis-ci.com/alan-turing-institute/simurgh.svg?branch=master)](https://travis-ci.com/alan-turing-institute/simurgh)
- Paper: [Brittain, M., and Pei, W.](https://arxiv.org/pdf/1905.01303.pdf)
- [Bluebird](https://github.com/alan-turing-institute/bluebird)
- [Twitcher](https://github.com/alan-turing-institutetwitcher)
- [Dodo](https://github.com/alan-turing-institute/dodo)
- [Aviary](https://github.com/alan-turing-institute/aviary)

