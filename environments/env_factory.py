import gymnasium as gym
from .random import RandomMap
from .baseline import Baseline

SEED = 42  # ask @Audrey
MAX_EPISODE_STEPS = 100  # ask @Audrey
# pas super propre de laisser ça là mais
# OK


class EnvFactory:

    _configs = {
        "random": RandomMap,
        "baseline": Baseline,
    }

    @staticmethod
    def create(env_name: str, render_mode: str = None) -> gym.Env:
        if env_name not in EnvFactory._configs:
            choices = list(EnvFactory._configs.keys())
            raise ValueError(f"Unknown environment. Available: {choices}")

        config = EnvFactory._configs[env_name]()

        if env_name == "random":
            config.seed = SEED

        env = gym.make(
            "FrozenLake-v1",  # shouldn't change as we created the whole program around this
            desc=config.desc,
            map_name=config.map_name,
            is_slippery=config.is_slippery,
            success_rate=config.success_rate,
            render_mode=render_mode,
            reward_schedule=config.reward_schedule,
            max_episode_steps=MAX_EPISODE_STEPS,  # ask @Audrey
        )

        return env
