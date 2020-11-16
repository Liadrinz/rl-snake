from environ import SnakeEnv

if __name__ == "__main__":
    env = SnakeEnv({"board_shape": [16, 16], "length": 5})
    while True:
        env.render()
        env.step(int(input()))
