import gymnasium as gym
from environments import DefaultConfig, RandomMap, CustomMap8x8

SEED = 42  # ask @Audrey
MAX_EPISODE_STEPS = 100  # ask @Audrey
# pas super propre de laisser ça là mais
# OK


class EnvFactory:

    _configs = {
        "default": DefaultConfig,
        "random": RandomMap,
        "custom8x8": CustomMap8x8,
    }

    @staticmethod
    def create(env_name: str, render_mode: str = None) -> gym.Env:
        if env_name not in EnvFactory._configs:
            raise ValueError("Unknown environment")

        config = EnvFactory._configs[env_name]()

        if env_name == "random":
            config.seed = SEED

        env = gym.make(
            "FrozenLake-v1",  # shouldn't change as we created the whole program around this
            desc=config.desc,
            map_name=config.map_name,
            is_slippery=config.is_slippery,
            render_mode=render_mode,
            # max_episode_steps= MAX_EPISODE_STEPS to add in our configs ? fix only one
            # ask @Audrey
        )
        return env
