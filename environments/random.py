from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class RandomMap:
    def __init__(self, size=8):
        self.desc = generate_random_map(size=size)
        self.map_name = None
        self.is_slippery = True
        self.success_rate = 0.95
        self.reward_schedule = (1, 0, 0)  # récompenses (victoire, trou, step)
