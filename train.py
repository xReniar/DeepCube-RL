from environment import Environment
from agents import SA2C, DQN, PPO
import yaml
from itertools import count
import torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.mps.is_available() else
    "cpu"
)

def train_DQN(env: Environment, agent: DQN):
    for episode in range(agent.num_episodes):
        state = env.reset()

        current_reward = env.algorithm.status(env.cube)
        for t in range(env.scramble_moves * 2):
            action = agent.action(state)
            obs, reward, done = env.step(action.item())

            current_reward = reward

            torch_current_reward = torch.tensor([current_reward], device=device)

            if done:
                next_state = None
            else:
                next_state = obs

            agent.memory.push(state, action, next_state, torch_current_reward)
            state = next_state

            agent.optimize()

            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*agent.tau + target_net_state_dict[key]*(1-agent.tau)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                with open(f"log/episode{episode}-{t}.txt", "w") as f:
                    f.write(env.cube.__str__())
                    f.write(env.start_state)
                break

def train_SA2C(env: Environment, agent: SA2C):
    t_so_far = 0
    i_so_far = 0

    while t_so_far < agent.total_timesteps:
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = agent.rollout()

if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    env = Environment(
        method="CFOP",
        size=3,
        device=device,
        args=args
    )

    train_DQN(env, DQN(args["DQN"]))
    #train_A2C(env, SA2C(args["A2C"]))