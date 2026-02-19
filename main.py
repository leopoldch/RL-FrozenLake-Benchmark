import argparse
from environments import EnvFactory
from strategies import StrategyFactory


def main():
    parser = argparse.ArgumentParser(description="RL FrozenLake")
    parser.add_argument("--env", type=str, default="random", help="Environment key")
    parser.add_argument("--strategy", type=str, default="random", help="Strategy key")
    parser.add_argument("--episodes", type=int, default=10, help="Amount of episodes")
    parser.add_argument("--render", action="store_true", help="Display the environment")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    # we created factories to easily test our environments and strategies
    # this way the main function is shared for all of our strategies and env
    env = EnvFactory.create(args.env, render_mode=render_mode)
    agent = StrategyFactory.create(
        args.strategy,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )

    print(f"Environment : {args.env}")
    print(f"Strategy     : {args.strategy} ({agent.__class__.__name__})")
    print(f"Episodes      : {args.episodes}")

    for episode in range(args.episodes):
        observation, info = env.reset()
        episode_over = False
        total_reward = 0
        step_count = 0

        while not episode_over:
            action = agent.select_action(observation, info)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.update(
                observation,
                action,
                reward,
                terminated,
                truncated,
                next_observation,
                info,
            )

            observation = next_observation
            total_reward += reward
            step_count += 1
            episode_over = terminated or truncated

            if terminated and reward == 0:  # we fell into a hole
                fell_in_hole = True
            else:
                fell_in_hole = False

        status = (
            "Found !" if total_reward > 0 else ("Hole" if fell_in_hole else "Not found")
        )
        print(f"Episode {episode + 1:03d} | Steps: {step_count:03d} | Result: {status}")

    env.close()


if __name__ == "__main__":
    main()
