class Baseline:
    def __init__(self):
        self.desc = [
            "SFFFFFFF",
            "FHFFFHFF",
            "FFFHFFFF",
            "FFFFFFHF",
            "FHFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFFFFFG",
        ]
        self.map_name = None
        self.is_slippery = True
        self.success_rate = 0.95  # 5% de glisse
        self.reward_schedule = (1, -1, 0)  # récompenses (victoire, trou, step)
