import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments import EnvFactory
from strategies.erm import ERMStrategy

ALPHA = 0.1
EPSILON = 0.1
BETA_C = 1e-10
REWARD_INF_NORM = 1.0

ENVS = ["baseline", "slippery"]
BETAS = [0.25, 0.5, 0.75, 1.0]
STEP_POWERS = [0.5, 0.6, 0.7]

BOUNDS_EPISODES = 1000
COARSE_EPISODES = 1000
COARSE_SEEDS = [0, 1, 2]
VAL_EPISODES = 3000
VAL_SEEDS = [0, 1, 2, 3, 4]
TOP_K = 3


def estimate_bounds(step_power):
    c_max, xmin, xmax = -np.inf, np.inf, -np.inf
    for env_name in ENVS:
        for seed in COARSE_SEEDS:
            random.seed(seed); np.random.seed(seed)
            env = EnvFactory.create(env_name)
            env.action_space.seed(seed)
            nS, nA = env.observation_space.n, env.action_space.n
            q = np.zeros((nS, nA))
            visits = np.zeros((nS, nA), dtype=np.int32)
            for ep in range(BOUNDS_EPISODES):
                obs, _ = env.reset(seed=seed) if ep == 0 else env.reset()
                done, G = False, 0.0
                while not done:
                    if np.random.random() < EPSILON:
                        a = env.action_space.sample()
                    else:
                        v = q[obs]
                        a = int(np.random.choice(np.flatnonzero(v == v.max())))
                    nxt, r, term, trunc, _ = env.step(a)
                    visits[obs, a] += 1
                    lr = ALPHA / (visits[obs, a] ** step_power)
                    boot = 0.0 if (term or trunc) else float(q[nxt].max())
                    z = r + boot - q[obs, a]
                    q[obs, a] -= lr * (math.exp(-BETA_C * z) - 1.0)
                    G += r; obs = nxt; done = term or trunc
                    c_max = max(c_max, float(q.max()))
                    xmin = min(xmin, G); xmax = max(xmax, G)
            env.close()
    d = ((xmax - xmin) ** 2) / 8.0
    return {"c": c_max, "xmin": xmin, "xmax": xmax, "d": d}


def derive_bounds(beta, c, d):
    span = max(abs(c - beta * d), abs(c))
    return -REWARD_INF_NORM - 2.0 * span, REWARD_INF_NORM + 2.0 * span


def run_erm(env_name, cfg, episodes, seeds):
    srs, frs, rws, stps, sstps, bvs = [], [], [], [], [], []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed)
        env = EnvFactory.create(env_name)
        env.action_space.seed(seed)
        agent = ERMStrategy(
            action_space=env.action_space,
            observation_space=env.observation_space,
            alpha=ALPHA, epsilon=EPSILON,
            beta=cfg["beta"], z_min=cfg["z_min"], z_max=cfg["z_max"],
            step_power=cfg["step_power"],
        )
        tiles = env.unwrapped.desc.ravel()
        succ, holes, rewards, steps, succ_steps = 0, 0, [], [], []
        for ep in range(episodes):
            obs, info = env.reset(seed=seed) if ep == 0 else env.reset()
            done, G, n = False, 0.0, 0
            while not done:
                a = agent.select_action(obs, info)
                nxt, r, term, trunc, info = env.step(a)
                agent.update(obs, a, r, term, trunc, nxt, info)
                G += r; n += 1; obs = nxt; done = term or trunc
                if term:
                    if tiles[nxt] == b"H": holes += 1
                    elif tiles[nxt] == b"G": succ += 1; succ_steps.append(n)
            rewards.append(G); steps.append(n)
        env.close()
        srs.append(succ / episodes * 100.0)
        frs.append(holes / episodes * 100.0)
        rws.append(float(np.mean(rewards)))
        stps.append(float(np.mean(steps)))
        sstps.append(float(np.mean(succ_steps)) if succ_steps else float("nan"))
        bvs.append(float(agent.bound_violations))
    return {
        "env": env_name,
        "success_rate": float(np.mean(srs)),
        "fall_rate": float(np.mean(frs)),
        "avg_reward": float(np.mean(rws)),
        "avg_steps": float(np.mean(stps)),
        "avg_success_steps": float(np.nanmean(sstps)),
        "bound_violations": float(np.mean(bvs)),
    }


def aggregate(summaries):
    s = float(np.mean([x["success_rate"] for x in summaries]))
    f = float(np.mean([x["fall_rate"] for x in summaries]))
    return {
        "mean_success": s,
        "mean_fall": f,
        "mean_reward": float(np.mean([x["avg_reward"] for x in summaries])),
        "mean_steps": float(np.mean([x["avg_steps"] for x in summaries])),
        "mean_success_steps": float(np.nanmean([x["avg_success_steps"] for x in summaries])),
        "mean_bound_violations": float(np.mean([x["bound_violations"] for x in summaries])),
        "ideal_distance": math.sqrt(((100.0 - s) / 100.0) ** 2 + (f / 100.0) ** 2),
    }


def evaluate(configs, bounds_cache, episodes, seeds):
    results = []
    for cfg in configs:
        summaries = [run_erm(e, cfg, episodes, seeds) for e in ENVS]
        results.append({
            "config": {k: cfg[k] for k in ("beta", "step_power", "z_min", "z_max")},
            "bounds_stats": bounds_cache[cfg["step_power"]],
            "env_summaries": summaries,
            "aggregate": aggregate(summaries),
        })
    results.sort(key=lambda r: (
        r["aggregate"]["ideal_distance"],
        -r["aggregate"]["mean_success"],
        r["aggregate"]["mean_fall"],
    ))
    return results


def main():
    bounds_cache = {sp: estimate_bounds(sp) for sp in STEP_POWERS}

    # grille beta x step_power
    configs = []
    for sp in STEP_POWERS:
        b = bounds_cache[sp]
        for beta in BETAS:
            zmin, zmax = derive_bounds(beta, b["c"], b["d"])
            configs.append({"beta": beta, "step_power": sp, "z_min": zmin, "z_max": zmax})

    # recherche et validation des top-k
    coarse = evaluate(configs, bounds_cache, COARSE_EPISODES, COARSE_SEEDS)
    top_cfgs = [r["config"] for r in coarse[:TOP_K]]
    validation = evaluate(top_cfgs, bounds_cache, VAL_EPISODES, VAL_SEEDS)

    print(json.dumps({
        "fixed_params": {"alpha": ALPHA, "epsilon": EPSILON, "beta_c": BETA_C},
        "calibration_envs": ENVS,
        "coarse_search": {
            "bounds_episodes": BOUNDS_EPISODES,
            "eval_episodes": COARSE_EPISODES,
            "seeds": COARSE_SEEDS,
            "results": coarse,
        },
        "validation": {
            "eval_episodes": VAL_EPISODES,
            "seeds": VAL_SEEDS,
            "results": validation,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
