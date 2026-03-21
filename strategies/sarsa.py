import numpy as np
from .strategy_interface import IStrategy


class SarsaStrategy(IStrategy):
    """
    Basé sur le cours IFT-7201 (Audrey Durand)
    """
    # on-policy
    # normalement plus prudent
    
    def __init__(
        self,
        action_space,
        observation_space=None,
        # hyper params à fixer @Audrey
        alpha=0.05,
        gamma=0.99,
        epsilon=0.05,
    ):
        super().__init__(action_space, observation_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        n_states = observation_space.n
        n_actions = action_space.n

        self.q = np.zeros((n_states, n_actions))

        # transition en attente (S_t, A_t, R_{t+1}, S_{t+1})
        self.pending = None

    def select_action(self, observation, info=None):
        if np.random.random() < self.epsilon:
            action = self.action_space.sample()  # exploration
        else:
            # pas de biais si égalité
            best_actions = np.flatnonzero(
                self.q[observation] == np.max(self.q[observation])
            )
            action = int(np.random.choice(best_actions))  # exploitation

        if self.pending is not None:
            state, prev_action, reward, next_state = self.pending
            self.q[state][prev_action] += self.alpha * (
                reward
                + self.gamma * self.q[next_state][action]
                - self.q[state][prev_action]
            )
            self.pending = None

        return action

    def update(
        self, observation, action, reward, terminated, truncated, next_observation, info
    ):
        if terminated:
            self.q[observation][action] += self.alpha * (
                reward - self.q[observation][action]
            )
            self.pending = None
        elif truncated:
            self.q[observation][action] += self.alpha * (
                reward - self.q[observation][action]
            )
            self.pending = None
        else:
            self.pending = (observation, action, reward, next_observation)
