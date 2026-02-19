from strategies import RandomStrategy

# we need to import manually our strategies here


class StrategyFactory:
    _strategies = {
        "random": RandomStrategy,
        # we can add more strategies here
    }

    @staticmethod
    def create(strategy_name: str, action_space, observation_space=None):
        if strategy_name not in StrategyFactory._strategies:
            choices = list(StrategyFactory._strategies.keys())
            raise ValueError(f"Unknown strategy.")

        return StrategyFactory._strategies[strategy_name](
            action_space=action_space, observation_space=observation_space
        )
