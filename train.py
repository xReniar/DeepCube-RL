from environment import Environment, generate_neighbors
from agents import A2C, DQN, PPO
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

def train_A2C(env: Environment, agent: A2C):
    for iter in range(agent.num_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        state = env.reset()

        for i in count():
            dist, value = agent.action(state)

            action = dist.sample()
            next_state, reward, done = env.step(action.item())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break
        
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = agent.critic_net(next_state)
        returns = agent.compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        agent.actor_optim.zero_grad()
        agent.critic_optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        agent.actor_optim.step()
        agent.critic_optim.step()


if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    env = Environment(
        method="LBL",
        size=3,
        device=device,
        args=args
    )

    #train_DQN(env, DQN(args["DQN"]))
    train_A2C(env, A2C(args["A2C"]))