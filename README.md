# Layer by layer method using Reinforcement Learning
This project aims to teach a neural network using reinforcement learning to learn how to solve the rubik's cube using the layer by layer method.

## Setup
This project uses `python3` and `nodejs`, check if they are installed:
```bash
python3 --version
node --version
```

## Create ground truth
To create the ground truth and format it correctly use this commands:
```bash
cd data/

# creates data.json
node create_data.js

# creates dataset.json
python3 format_data.py
```

## How to use
First install the requirements:
```bash
pip install -r requirements.txt
```

This project creates environment only for `cross` and `f2l` steps. After training a `phase.pth` will be saved inside `./models/DQN/`. To train run `train.py` with the following args:
- `--phase`: this argument is mandatory and specifies the phase to train, the possible values are `cross` and `f2l`
- `--load-experience`: this argument specifies whether to load moves from an online solver.

The `./data/dataset.json` is created using [this](https://solverubikscube.com/) solver