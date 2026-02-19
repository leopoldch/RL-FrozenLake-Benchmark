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
