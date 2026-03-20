class Dangerous:
    def __init__(self):
        self.desc = [
            "SFFFFFFF",
            "FHFHFFHF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FFFHFFHF",
            "FHFFFHFF",
            "FFFHFFFF",
            "FFHFFFHG",
        ]
        self.map_name = None
        self.is_slippery = True
        self.success_rate = 0.70  # 30% de glisse
        self.reward_schedule = (1, 0, 0)