from strategies import IStrategy


class RandomStrategy(IStrategy):
    def select_action(self, observation, info=None):
        return self.action_space.sample()  # Joue au hasard
