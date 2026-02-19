class DefaultConfig:
    def __init__(self):
        self.desc = None
        self.map_name = "4x4"
        self.is_slippery = True
        self.success_rate = 1.0 / 3.0
        self.reward_schedule = (1, 0, 0)
