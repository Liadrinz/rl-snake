import argparse
import gym
import ray
from ray.rllib.agents.dqn import DQNTrainer

from environ import SnakeEnv

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--restore", default=None)

def train_one_step():
    result = agent.train()
    fields = ["episode_reward_max", "episode_reward_min", "episode_reward_mean", "episode_len_mean"]
    print(", ".join(["{}: {}".format(k, result[k]) for k in result if k in fields]))

def save_ckpt():
    ckpt_path = agent.save()
    print("saved to {}".format(ckpt_path))

def simulate_one_game(render=False):
    score = 0
    obs = snake_env.reset()
    while True:
        if render: snake_env.render()
        action = agent.compute_action(obs)
        obs, r, done, _ = snake_env.step(action)
        if r == 1.0:
            score += 1
        if done:
            snake_env.reset()
            break
    return score

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_gpus=1)
    env_config = {"board_shape": [8, 8], "length": 3}
    config = {
        "env": SnakeEnv,
        "env_config": env_config,
        "num_gpus": 1,
        "lr": 1e-4,
        "hiddens": [32, 64, 512]
    }
    agent = DQNTrainer(config=config)
    snake_env = SnakeEnv(config=env_config)
    if args.test:
        assert args.restore is not None
        agent.restore(args.restore)
        while True:
            score = simulate_one_game(render=True)
            print("Score: {}".format(score))
    else:
        if args.restore is not None:
            agent.restore(args.restore)
            i = agent.iteration
        else:
            i = 0
        while True:
            train_one_step()
            if i % 10 == 0:
                save_ckpt()
                # avg_score = 0
                # for _ in range(100):
                #     avg_score += simulate_one_game() / 100
                # print("Avg score: {}".format(avg_score))
            i += 1
    ray.shutdown()
