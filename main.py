import argparse
from environments import EnvFactory
from strategies import StrategyFactory
import random
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="RL FrozenLake")
    parser.add_argument("--env", type=str, default="random", help="Environment key")
    parser.add_argument("--strategy", type=str, default="random", help="Strategy key")
    parser.add_argument("--episodes", type=int, default=10, help="Amount of episodes")
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of iterations"
    )
    parser.add_argument("--render", action="store_true", help="Display the environment")
    parser.add_argument(
        "--plot", action="store_true", help="Display metrics after training"
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    def run_iteration(seed):
        random.seed(seed)
        np.random.seed(seed)
        # we created factories to easily test our environments and strategies
        # this way the main function is shared for all of our strategies and env
        env = EnvFactory.create(args.env, render_mode=render_mode)
        agent = StrategyFactory.create(
            args.strategy,
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        env.action_space.seed(seed)

        successes = 0
        rewards_per_episode = []
        holes_per_episode = []
        successes_per_episode = []
        steps_per_episode = []

        for episode in range(args.episodes):
            if episode == 0:
                observation, info = env.reset(seed=seed)
            else:
                observation, info = env.reset()
            episode_over = False
            total_reward = 0
            step_count = 0
            fell_in_hole = False

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

            success = total_reward > 0
            if success:
                successes += 1
            rewards_per_episode.append(total_reward)
            holes_per_episode.append(int(fell_in_hole))
            successes_per_episode.append(int(success))
            steps_per_episode.append(step_count)

        env.close()

        success_rate = successes / args.episodes * 100
        avg_reward = sum(rewards_per_episode) / args.episodes
        avg_steps = sum(steps_per_episode) / args.episodes

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "total_holes": sum(holes_per_episode),
        }

    print(f"Environment : {args.env}")
    print(f"Strategy     : {args.strategy}")
    print(f"Episodes      : {args.episodes}")
    print(f"Iterations    : {args.iterations}")
    print()

    iteration_results = []
    for i in range(args.iterations):
        seed = i
        # print de debug pendant l'éxécution
        print(f"--- Iteration {i + 1}/{args.iterations} (seed={seed}) ---")
        result = run_iteration(seed)
        iteration_results.append(result)
        print(f"  Success rate : {result['success_rate']:.1f}%")
        print(f"  Avg reward   : {result['avg_reward']:.3f}")
        print(f"  Avg steps    : {result['avg_steps']:.1f}")
        print(f"  total holes  : {result['total_holes']}")
        print()

    if args.iterations > 1:
        avg_sr = sum(r["success_rate"] for r in iteration_results) / args.iterations
        avg_rw = sum(r["avg_reward"] for r in iteration_results) / args.iterations
        avg_st = sum(r["avg_steps"] for r in iteration_results) / args.iterations
        avg_holes = sum(r["total_holes"] for r in iteration_results) / args.iterations
        print(f"=== Moyenne sur {args.iterations} itérations ===")
        print(f"  Success rate : {avg_sr:.1f}%")
        print(f"  Avg reward   : {avg_rw:.3f}")
        print(f"  Avg steps    : {avg_st:.1f}")
        print(f"  Avg holes    : {avg_holes:.1f}")

    # if args.plot:
    #    plot_training_results()


if __name__ == "__main__":
    main()
