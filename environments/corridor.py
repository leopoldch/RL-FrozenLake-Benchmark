class Corridor:
    def __init__(self):
        self.desc = [
            "SFFFFFFF",
            "FFFFFFFF",
            "HHHHHHFF",
            "FFFFHFFF",
            "FFFFHFFF",
            "FFFFHFFF",
            "FFFFHFFF",
            "FFFFFHFG",
        ]
        self.map_name = None
        self.is_slippery = True
        self.success_rate = 0.70
        self.reward_schedule = (1, -1, 0)