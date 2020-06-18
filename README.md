# ATC-RL
This project is about application of reinforcement learning to real time system control. The goal is to train a neural network to perform basic task to control real system through reinforcement learning. 

The case study that is currently being worked on is as follows:

In practice, air traffic control operators (ATCOs) monitor a given sector and issue commands to aircraft pilots to ensure safe separation between aircraft. They also have to consider the number and frequency of instructions issued, fuel efficiency and orderly handover between sectors. Optimising for the multiple objectives while accounting for uncertainty (e.g., due to aircraft mass, pilot behaviour or weather conditions) makes this a particularly complex task.

### Sample performance with random starting locations:
The RL algorithm is tested for 1000 different random locations to check the model performance for unseen scenarios. The number of crashes were around 90, but the performance was not that bad given the training time. This can certainly be improved by modifying the parameters and adding some more action variables.

<!--- ![sample_gif](https://github.com/arunbalas/ATC-RL/blob/develop/Final%20Gif.gif) -->

<p align="center">
  <img width="460" height="300" src="https://github.com/arunbalas/ATC-RL/blob/develop/Final%20Gif.gif">
</p>

In addition, this approach could improve throughput of a sector, noise abatement and increase efficiency through continuous 
climb and descend profiles, which for example could save 1-2% of fuel. 

References:
[![DOI](https://zenodo.org/badge/148370950.svg)](https://zenodo.org/badge/latestdoi/148370950)
[![DOI](https://travis-ci.com/alan-turing-institute/simurgh.svg?branch=master)](https://travis-ci.com/alan-turing-institute/simurgh)
- Paper: [Brittain, M., and Pei, W.](https://arxiv.org/pdf/1905.01303.pdf)
- [Bluebird](https://github.com/alan-turing-institute/bluebird)
- [Twitcher](https://github.com/alan-turing-institutetwitcher)
- [Dodo](https://github.com/alan-turing-institute/dodo)
- [Aviary](https://github.com/alan-turing-institute/aviary)

## Setup
Clone repository:
```bash
git clone --branch develop https://github.com/arunbalas/ATC-RL.git
```

## Requisites & Installation

Please refer to requirements.txt file.

```bash
pip install -r requirements.txt
```


### Dependencies

- Need Bluesky, Bluebird and Pydodo which can be easily installed by following the steps given below:

### 1. Clone this repository


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

If BlueSky and BlueBird are running (see previous step), then one can communicate with the simulator (via
BlueBird) using PyDodo:

```python
>>> import pydodo
>>>
>>> pydodo.reset_simulation()
True
>>>
```

## Train Agent

Please start respective single and multi-agent Jupyter notebooks.



