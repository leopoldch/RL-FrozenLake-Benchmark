from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class DefaultConfig:
    def __init__(self):
        self.desc = None
        self.map_name = "4x4"
        self.is_slippery = True
        self.success_rate = 1.0 / 3.0
        self.reward_schedule = (1, 0, 0)


class RandomMap:
    def __init__(self):
        self.desc = generate_random_map(size=8)
        # we don't care about the others for now
        self.map_name = None
        self.is_slippery = True
        self.success_rate = 1.0 / 3.0
        self.reward_schedule = (1, 0, 0)


class CustomMap8x8:
    def __init__(self):
        self.desc = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
        self.map_name = None  # no need as we defined the map ourselves
        self.is_slippery = True
        self.success_rate = 1.0 / 3.0
        self.reward_schedule = (1, 0, 0)
