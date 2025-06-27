import torch
import torch.nn as nn
import numpy as np
from collections import deque
from cube import Cube


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent():
    def __init__(self):
        self.n_solve = 0
        self.epsilon = 0 # randomness?
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)

    def get_state(self, cube):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass
    
    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(sekf, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    cube = Cube()

    while True:
        # get old states
        state_old = agent.get_state(cube)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = cube.play_step(final_move)
        state_new = agent.get_state(cube)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            cube.reset()
            agent.n_solve += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model_save()

            print(f"Solve: {agent.n_solve}, Score: {score}, Record: {record}")

