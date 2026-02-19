from abc import ABC, abstractmethod

# import pour définir une classe abstraite


class IStrategy(ABC):
    def __init__(self, action_space, observation_space=None):
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def select_action(self, observation, info=None):
        pass

    def update(
        self, observation, action, reward, terminated, truncated, next_observation, info
    ):
        pass
