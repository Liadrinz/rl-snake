from typing import List, Tuple
import gym
import numpy as np

from gym.spaces import Discrete, Box
from renderer import SnakeRenderer

class SnakeEnv(gym.Env):

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 2
    }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.bx, self.by = config["board_shape"]
        self._full_set = set([(x,y) for x in range(self.bx) for y in range(self.by)])
        self.snake = [np.array([self.bx // 2 + i, self.by // 2]).astype(int) for i in range(self.config["length"])]
        self.food = None
        self.food = self.random_food()
        self.direction = 0  # 0-3: 左下右上
        self._direction_map = np.array([[-1,0],[0,-1],[1,0],[0,1]]).astype(int)
        self.action_space = Discrete(5)  # 0-4: 无左下右上
        self.observation_space = Box(0, 2, shape=config["board_shape"], dtype=np.float32)
        self.renderer = SnakeRenderer(self)

    def reset(self) -> List:
        self.snake = [np.array([self.bx // 2 + i, self.by // 2]).astype(int) for i in range(self.config["length"])]
        self.food = self.random_food()
        self.direction = 0
        return self.parse_state()
    
    def step(self, action: int) -> Tuple[List, float, bool, dict]:
        assert action in [0, 1, 2, 3, 4]
        n_direction = action - 1
        if action != 0 and abs(n_direction - self.direction) != 2:
            self.direction = n_direction
        r, done = self.move_snake()
        return self.parse_state(), r, done, {}
    
    def move_snake(self) -> Tuple[float, bool]:
        self.snake.pop()
        self.snake.insert(0, self.snake[0] + self._direction_map[self.direction])
        # 吃到食物
        if (self.snake[0] == self.food).all():
            self.snake.insert(0, self.food + self._direction_map[self.direction])
            self.food = self.random_food()
            return 1.0, False
        # 撞墙(越界)
        if not ((0 <= self.snake[0][0] < self.bx) and (0 <= self.snake[0][1] < self.by)):
            return -10.0, True
        # 撞到自己
        if (self.snake[0] == self.snake[1:]).all(axis=1).any():
            return -10.0, True
        return -0.1, False
    
    def random_food(self):
        choices_available = self._full_set.difference(set([(x, y) for x, y in self.snake]))
        if self.food is not None:
            choices_available = choices_available.difference(set([(*self.food,)]))
        choice_idx = np.random.choice(np.arange(len(choices_available)))
        return np.array(list(choices_available)[choice_idx])
    
    def parse_state(self) -> List:
        space = np.zeros(self.config["board_shape"])
        space[self.food] = 2.0
        for pos in self.snake:
            if not ((0 <= pos[0] < self.bx) and (0 <= pos[1] < self.by)):
                continue
            space[pos] = 1.0
        return space

    def render(self, mode="human"):
        return self.renderer.render(mode)
    
    def close(self):
        return self.renderer.close()