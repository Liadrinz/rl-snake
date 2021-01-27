from typing import List, Tuple
import gym
import numpy as np

from gym import spaces
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
        self.action_space = spaces.Discrete(5)  # 0-4: 无左下右上
        # self.observation_space = Box(0, 1, shape=(self.bx, self.by), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "food": spaces.Box(low=1, high=max(self.bx, self.by) + 1, shape=(2,)),
            "snake": spaces.Box(low=0, high=max(self.bx, self.by) + 1, shape=(self.bx * self.by, 2))
        })
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
    
    def potential(self, s1, s2) -> float:
        dist1 = np.abs(s1["snake"][0] - s1["food"])
        dist2 = np.abs(s2["snake"][0] - s2["food"])
        return np.sum(dist1 - dist2) / (self.bx + self.by)

    def move_snake(self) -> Tuple[float, bool]:
        s1 = self.parse_state()
        self.snake.pop()
        self.snake.insert(0, self.snake[0] + self._direction_map[self.direction])
        s2 = self.parse_state()
        potential = self.potential(s1, s2)
        # 吃到食物
        if (self.snake[0] == self.food).all():
            self.snake.insert(0, self.food + self._direction_map[self.direction])
            self.food = self.random_food()
            return 1.0 + potential, False
        # 撞墙(越界)
        if not ((0 <= self.snake[0][0] < self.bx) and (0 <= self.snake[0][1] < self.by)):
            return -1000.0, True
        # 撞到自己
        if (self.snake[0] == self.snake[1:]).all(axis=1).any():
            return -1000.0, True
        return potential, False
    
    def random_food(self):
        choices_available = self._full_set.difference(set([(x, y) for x, y in self.snake]))
        if self.food is not None:
            choices_available = choices_available.difference(set([(*self.food,)]))
        choice_idx = np.random.choice(np.arange(len(choices_available)))
        return np.array(list(choices_available)[choice_idx])
    
    def parse_state(self) -> List:
        # flat_vec = np.array([self.by, 1]).astype(int)
        # space = np.zeros((self.bx * self.by, ))
        # space[np.sum(self.food * flat_vec)] = 0.5
        # indices = np.sum(self.snake * flat_vec, axis=1).tolist()
        # indices.sort()
        # pivot = self._pivot(indices, self.bx * self.by, 0, len(indices))
        # indices = indices[:pivot]
        # space[indices] = 0.5
        # head = np.sum(self.snake[0] * flat_vec)
        # if head < self.bx * self.by:
        #     space[head] = 1.0
        # return space.reshape((self.bx, self.by))
        food = self.food + 1
        snake = np.array(self.snake) + 1
        pads = np.zeros((self.bx * self.by - snake.shape[0], 2))
        snake = np.concatenate((snake, pads), axis=0)
        return {"food": food, "snake": snake}

    def render(self, mode="human"):
        return self.renderer.render(mode)
    
    def close(self):
        return self.renderer.close()
    
    def _dist(self):
        return np.sum(np.abs(self.food - self.snake[0]))
    
    def _pivot(self, sorted_list, v, l, r):
        m = (l + r) // 2
        if r - l <= 1:
            if v <= sorted_list[m]:
                return l
            else:
                return r
        if v <= sorted_list[m]:
            return self._pivot(sorted_list, v, l, m)
        else:
            return self._pivot(sorted_list, v, m, r)
