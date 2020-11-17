from gym.envs.classic_control import rendering

class SnakeRenderer:

    def __init__(self, env, scale=20) -> None:
        self.env = env
        self.bx, self.by = self.env.config["board_shape"]
        self.scale = scale
        self.viewer = None
        self.snake_geoms = []
        self.food_geom = None
    
    def init_snake(self):
        for snake_part in self.env.snake:
            rect = rendering.make_polygon(v=self.get_square(*snake_part))
            self.snake_geoms.append(rect)
            self.viewer.add_geom(rect)

    def init_food(self):
        rect = rendering.make_polygon(v=self.get_square(*self.env.food))
        self.food_geom = rect
        self.viewer.add_geom(rect)
    
    def get_square(self, gx, gy):
        rx, ry = gx * self.scale, gy * self.scale
        points = [
            (rx, ry),
            (rx + self.scale, ry),
            (rx + self.scale, ry + self.scale),
            (rx, ry + self.scale)]
        return points

    def render(self, mode="human"):
        if self.viewer == None:
            self.viewer = rendering.Viewer(self.bx * self.scale, self.by * self.scale)
            self.init_snake()
            self.init_food()
        self.food_geom.v = self.get_square(*self.env.food)
        i = 0
        while i < len(self.env.snake):
            square = self.get_square(*self.env.snake[i])
            if i < len(self.snake_geoms):
                self.snake_geoms[i].v = self.get_square(*self.env.snake[i])
            else:
                rect = rendering.make_polygon(v=square)
                self.snake_geoms.append(rect)
                self.viewer.add_geom(rect)
            i += 1
        while i < len(self.viewer.geoms):
            target = self.viewer.geoms[i]
            if target == self.food_geom:
                i += 1
            else:
                self.viewer.geoms.remove(target)
        return self.viewer.render(mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None