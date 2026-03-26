import argparse
import os
import random

import numpy as np

from environments import EnvFactory
from strategies import StrategyFactory
from utils import plot_training_results, plot_policy


def main():
    parser = argparse.ArgumentParser(description="RL FrozenLake")
    parser.add_argument("--env", type=str, default="random", help="Environment key")
    parser.add_argument("--strategy", type=str, default="random", help="Strategy key")
    parser.add_argument("--episodes", type=int, default=60000, help="Amount of episodes")
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of iterations"
    )
    parser.add_argument("--render", action="store_true", help="Display the environment")
    parser.add_argument(
        "--plot", action="store_true", help="Display metrics after training"
    )
    parser.add_argument("--window", type=int, default=200, help="Smoothing window size")
    parser.add_argument(
        "--save-dir", type=str, default="figures", help="Directory to save figures"
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
        state_tiles = env.unwrapped.desc.ravel()

        successes = 0
        rewards_per_episode = []
        holes_per_episode = []
        successes_per_episode = []
        steps_per_episode = []
        successful_steps_per_episode = []

        for episode in range(args.episodes):
            if episode == 0:
                observation, info = env.reset(seed=seed)
            else:
                observation, info = env.reset()
            episode_over = False
            total_reward = 0
            step_count = 0
            fell_in_hole = False
            reached_goal = False

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

                #détecte la fin via la case atteinte, pas via la récompense!
                terminal_tile = state_tiles[next_observation]
                if terminated and terminal_tile == b"H":
                    fell_in_hole = True
                elif terminated and terminal_tile == b"G":
                    reached_goal = True

            success = reached_goal
            if success:
                successes += 1
            rewards_per_episode.append(total_reward)
            holes_per_episode.append(int(fell_in_hole))
            successes_per_episode.append(int(success))
            steps_per_episode.append(step_count)
            successful_steps_per_episode.append(step_count if success else np.nan)

        desc = env.unwrapped.desc
        env.close()

        success_rate = successes / args.episodes * 100
        avg_reward = sum(rewards_per_episode) / args.episodes
        avg_steps = sum(steps_per_episode) / args.episodes

        return {
            "seed": seed,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "total_holes": sum(holes_per_episode),
            "rewards_per_episode": rewards_per_episode,
            "holes_per_episode": holes_per_episode,
            "successes_per_episode": successes_per_episode,
            "steps_per_episode": steps_per_episode,
            "successful_steps_per_episode": successful_steps_per_episode,
            "final_q": agent.q.copy() if hasattr(agent, "q") else None,
            "desc": desc,
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
        print(f"  Total holes  : {result['total_holes']}")
        print()

    # résumé avec IC95
    n = args.iterations
    sr = np.array([r["success_rate"] for r in iteration_results])
    rw = np.array([r["avg_reward"] for r in iteration_results])
    st = np.array([r["avg_steps"] for r in iteration_results])
    hl = np.array([r["total_holes"] for r in iteration_results])

    def format_interval(values):
        mean = np.mean(values)
        if n > 1:
            ci = 1.96 * np.std(values, ddof=1) / np.sqrt(n)
            return f"{mean:.2f} ± {ci:.2f}"
        return f"{mean:.2f}"

    print(f"=== Résumé sur {n} itération(s) ===")
    print(f"  Success rate : {format_interval(sr)}%")
    print(f"  Avg reward   : {format_interval(rw)}")
    print(f"  Avg steps    : {format_interval(st)}")
    succ_steps = np.array([np.nanmean(r["successful_steps_per_episode"]) for r in iteration_results])
    print(f"  Avg steps (success) : {format_interval(succ_steps)}")
    print(f"  Total holes  : {format_interval(hl)}")
    fall_pct = hl / args.episodes * 100
    print(f"  Fall rate    : {format_interval(fall_pct)}%")

    if args.plot:
        os.makedirs(args.save_dir, exist_ok=True)

        plot_training_results(
            iteration_results=iteration_results,
            strategy=args.strategy,
            env=args.env,
            window=args.window,
            save_path=os.path.join(
                args.save_dir, f"training_{args.env}_{args.strategy}.png"
            ),
        )

        # policy plot : on choisit le run le plus proche de la médiane
        median_sr = np.median(sr)
        best_idx = int(np.argmin(np.abs(sr - median_sr)))
        best_run = iteration_results[best_idx]

        if best_run["final_q"] is not None:
            plot_policy(
                q_table=best_run["final_q"],
                desc=best_run["desc"],
                title=f"Policy {args.strategy} / {args.env} (seed={best_run['seed']})",
                save_path=os.path.join(
                    args.save_dir, f"policy_{args.env}_{args.strategy}.png"
                ),
            )


if __name__ == "__main__":
    main()
