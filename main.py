import argparse
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer

from environ import SnakeEnv
from model import CustomModel

parser = argparse.ArgumentParser()
parser.add_argument("--phase", default="train")
parser.add_argument("--restore", default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    env_config = {"board_shape": [16, 16], "length": 5}
    config = {
        "env": SnakeEnv,
        "env_config": env_config,
        "num_gpus": 1,
        "model": {
            "custom_model": "my_model"
        },
        "vf_share_layers": True,
        "lr": 1e-4,
        "framework": "tf"
    }
    agent = PPOTrainer(config)
    if args.phase == "test":
        assert args.restore is not None
        snake_env = SnakeEnv(config=env_config)
        agent.restore(args.restore)
        obs = snake_env.parse_state()
        while True:
            snake_env.render()
            action = agent.compute_action(obs)
            obs, r, done, _ = snake_env.step(action)
            if done:
                snake_env.reset()
                obs = snake_env.parse_state()
    else:
        if args.restore is not None:
            agent.restore(args.restore)
        i = 0
        while True:
            result = agent.train()
            print(result)
            if i % 10 == 0:
                ckpt_path = agent.save()
                print("saved to {}".format(ckpt_path))
            i += 1
    ray.shutdown()
