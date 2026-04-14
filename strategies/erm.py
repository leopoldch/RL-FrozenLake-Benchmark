import numpy as np

from .strategy_interface import IStrategy


class ERMStrategy(IStrategy):
    """
    https://openreview.net/pdf?id=Ak4tP0vvna
    ERM Q-learning très simple adapté de Su et al. (NeurIPS 2025).
    """

    def __init__(
        self,
        action_space,
        observation_space=None,
        alpha=0.1,
        epsilon=0.1,
        beta=1.0, #un seul niveau de risque beta: equivalent au papier quand l'ensemble B est un singleton
        z_min=-2.0,
        z_max=2.0,
        step_power=0.5,
    ):
        super().__init__(action_space, observation_space)
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.z_min = z_min
        self.z_max = z_max
        self.step_power = step_power

        n_states = observation_space.n
        n_actions = action_space.n
        self.q = np.zeros((n_states, n_actions))
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int32)
        self.bound_violations = 0

    def select_action(self, observation, info=None):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        state_values = self.q[observation]
        finite_mask = np.isfinite(state_values)
        if not finite_mask.any():
            return self.action_space.sample()

        best_value = np.max(state_values[finite_mask])
        best_actions = np.flatnonzero(
            finite_mask & np.isclose(state_values, best_value)
        )
        return int(np.random.choice(best_actions))

    def _bootstrap_value(self, next_observation):
        next_values = self.q[next_observation]
        finite_mask = np.isfinite(next_values)
        if not finite_mask.any():
            return 0.0
        return float(np.max(next_values[finite_mask]))

    def update(
        self, observation, action, reward, terminated, truncated, next_observation, info
    ):
        # echantillonnage online episode par episode, comme les autres strategies
        if not np.isfinite(self.q[observation, action]):
            return

        bootstrap = 0.0
        if not (terminated or truncated):
            bootstrap = self._bootstrap_value(next_observation)

        # z = reward + bootstrap - q  (ref au résidus TD)
        td_residual = reward + bootstrap - self.q[observation, action]
        # pas de clipping : si la borne est violee, on marque la valeur comme
        # non bornee (-inf), donc plus proche de l'Algorithm 1 du papier
        if not (self.z_min <= td_residual <= self.z_max):
            self.q[observation, action] = -np.inf
            self.bound_violations += 1
            return

        self.visit_counts[observation, action] += 1
        step_size = self.alpha / (
            self.visit_counts[observation, action] ** self.step_power
        )

        # update principale du papier :  q <- q - alpha * (exp(-beta * z) - 1)
        self.q[observation, action] -= step_size * (
            np.exp(-self.beta * td_residual) - 1.0
        )
