import numpy as np
from .strategy_interface import IStrategy


class QLearningStrategy(IStrategy):
    """
    Basé sur le cours IFT-7201 (Audrey Durand)
    """
    # off-policy

    def __init__(
        self,
        action_space,
        observation_space=None,
        # hyper params 
        alpha=0.1,
        gamma=1,
        epsilon=0.1,
    ):
        super().__init__(action_space, observation_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        n_states = observation_space.n
        n_actions = action_space.n

        self.q = np.zeros((n_states, n_actions)) # init à 0

    def select_action(self, observation, info=None):
        if np.random.random() < self.epsilon:
            action = self.action_space.sample()  # exploration
        else:
            # pas de biais si égalité
            best_actions = np.flatnonzero(
                self.q[observation] == np.max(self.q[observation])
            )
            action = int(np.random.choice(best_actions))  # exploitation

        return action

    def update(
        self, observation, action, reward, terminated, truncated, next_observation, info
    ):
        if terminated or truncated:
            self.q[observation][action] += self.alpha * (
                reward - self.q[observation][action]
            )
        else:
            self.q[observation][action] += self.alpha * (
                reward
                + self.gamma * np.max(self.q[next_observation])
                - self.q[observation][action]
            )
