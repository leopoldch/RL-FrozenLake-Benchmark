class Baseline:
    def __init__(self):
        self.desc = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFFF",
            "FHFFHFFF",
            "FFFHFFFG",
        ]
        self.map_name = None
        self.is_slippery = True
        self.success_rate = 0.95
        self.reward_schedule = (1, 0, 0)  # récompenses (victoire, trou, step)
