# Layer by layer method using Reinforcement Learning
This project aims to teach a neural network using reinforcement learning to learn how to solve the rubik's cube using the layer by layer method.

## How to use
First install the requirements:
```py
pip install -r requirements.txt
```
This project creates environment only for `cross` and `f2l` steps. After training a `phase.pth` will be saved inside `./models/DQN/`. To train run `train.py` with the following args:
- `--phase`: this argument is mandatory and specifies the phase to train, the possible values are `cross` and `f2l`
- `--load-experience`: this argument specifies whether to load moves from an online solver.

The `./data/dataset.json` is created using [this](https://solverubikscube.com/) solver