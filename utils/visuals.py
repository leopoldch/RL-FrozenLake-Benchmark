import numpy as np
import matplotlib.pyplot as plt

# visualisation des métriques

# taux de succès
# taux de chute dans les trous
# nombre cumulé de chutes
# longueur moyenne des épisodes réussis

# @Audrey t'es OK avec ça ?


def _rolling_mean(values: list, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _rolling_mean_successful_steps(
    steps: list, successes: list, window: int
) -> np.ndarray:
    steps_arr = np.array(steps)
    success_arr = np.array(successes)
    weighted_sum = np.convolve(steps_arr * success_arr, np.ones(window), mode="valid")
    success_count = np.convolve(success_arr, np.ones(window), mode="valid")
    with np.errstate(invalid="ignore"):
        return np.where(success_count > 0, weighted_sum / success_count, np.nan)


def _episodes_to_threshold(
    success_curve: np.ndarray, x: np.ndarray, threshold: float
) -> int | None:
    indices = np.where(success_curve >= threshold)[0]
    return int(x[indices[0]]) if len(indices) > 0 else None


def plot_training_results(
    rewards: list,
    holes: list,
    successes: list,
    steps: list,
    window: int = 100,
    strategy: str = "",
    env: str = "",
    threshold: float = 0.75,
) -> None:
    n_episodes = len(rewards)
    effective_window = min(window, n_episodes)

    success_curve = _rolling_mean(successes, effective_window)
    hole_curve = _rolling_mean(holes, effective_window)
    cumulative_holes = np.cumsum(holes)
    success_steps_curve = _rolling_mean_successful_steps(
        steps, successes, effective_window
    )

    x = np.arange(effective_window - 1, n_episodes)

    final_success_rate = float(np.mean(successes[-effective_window:]))
    final_hole_rate = float(np.mean(holes[-effective_window:]))
    ep_threshold = _episodes_to_threshold(success_curve, x, threshold)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Stratégie : {strategy}  |  Env : {env}  |  {n_episodes} épisodes  |  fenêtre = {effective_window}",
        fontsize=12,
    )

    axes[0, 0].plot(x, success_curve, color="steelblue", linewidth=1.2)
    axes[0, 0].axhline(
        threshold,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"seuil {threshold:.0%}",
    )
    if ep_threshold is not None:
        axes[0, 0].axvline(
            ep_threshold,
            color="green",
            linestyle=":",
            linewidth=1.2,
            label=f"atteint à l'ép. {ep_threshold}",
        )
    axes[0, 0].set_title(f"Taux de succès  (final : {final_success_rate:.1%})")
    axes[0, 0].set_ylabel("Taux")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, hole_curve, color="firebrick", linewidth=1.2)
    axes[0, 1].set_title(
        f"Taux de chute dans les trous  (final : {final_hole_rate:.1%})"
    )
    axes[0, 1].set_ylabel("Taux")
    axes[0, 1].set_ylim(0, 1)

    x_full = np.arange(n_episodes)
    axes[1, 0].plot(x_full, cumulative_holes, color="darkorange", linewidth=1.2)
    axes[1, 0].set_title(
        f"Chutes cumulées dans les trous  (total : {int(cumulative_holes[-1])})"
    )
    axes[1, 0].set_ylabel("Nb de chutes")

    axes[1, 1].plot(x, success_steps_curve, color="mediumpurple", linewidth=1.2)
    axes[1, 1].set_title("Longueur moyenne des épisodes réussis")
    axes[1, 1].set_ylabel("Steps")

    for ax in axes.flat:
        ax.set_xlabel("Épisode")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
