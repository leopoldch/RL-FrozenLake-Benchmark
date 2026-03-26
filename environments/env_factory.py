import gymnasium as gym
from .random import RandomMap
from .baseline import Baseline
from .slippery import Slippery
from .corridor import Corridor
from .riskDilemma import RiskDilemma

MAX_EPISODE_STEPS = 80  
# On peut le laisser à 80 si on garde des grilles 8x8
# car 14 pas minimum pr atteindre la cible 
# 80/14 = a peu pres la liberté de l'agent
# 80/14 = 5.7 environ et donc liberté de faire 
# 5.7 fois la distance minimum 


class EnvFactory:

    _configs = {
        "random": RandomMap,
        "baseline": Baseline,
        "slippery": Slippery,
        "corridor": Corridor,
        "riskDilemma": RiskDilemma
    }

    @staticmethod
    def create(env_name: str, render_mode: str = None) -> gym.Env:
        if env_name not in EnvFactory._configs:
            choices = list(EnvFactory._configs.keys())
            raise ValueError(f"Unknown environment. Available: {choices}")

        config = EnvFactory._configs[env_name]()

        if env_name == "random":
            config.seed = 42 # should not be used in our project 

        env = gym.make(
            "FrozenLake-v1",  # shouldn't change as we created the whole program around this
            desc=config.desc,
            map_name=config.map_name,
            is_slippery=config.is_slippery,
            success_rate=config.success_rate,
            render_mode=render_mode,
            reward_schedule=config.reward_schedule,
            max_episode_steps=MAX_EPISODE_STEPS,
        )

        return env
