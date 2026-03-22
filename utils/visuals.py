import numpy as np
import matplotlib.pyplot as plt

# visualisation des métriques

# taux de succès
# taux de chute dans les trous
# nombre cumulé de chutes
# longueur moyenne des épisodes réussis

# @Audrey t'es OK avec ça ?


def moving_average(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def moving_nanmean(x, window):
    n = len(x) - window + 1
    out = np.empty(n)
    with np.errstate(all="ignore"):
        for i in range(n):
            out[i] = np.nanmean(x[i : i + window])
    return out


def mean_ci95(curves):
    with np.errstate(all="ignore"):
        mean = np.nanmean(curves, axis=0)
        if curves.shape[0] < 2:
            return mean, np.zeros_like(mean)
        std = np.nanstd(curves, axis=0, ddof=1)
        ci = 1.96 * std / np.sqrt(curves.shape[0])
    return mean, ci


def plot_training_results(iteration_results, strategy, env, window, save_path=None):
    n_runs = len(iteration_results)
    n_episodes = len(iteration_results[0]["successes_per_episode"])
    window = max(1, min(window, n_episodes))

    # matrices (n_runs, n_episodes)
    success_mat = np.array([r["successes_per_episode"] for r in iteration_results])
    holes_mat = np.array([r["holes_per_episode"] for r in iteration_results])
    steps_mat = np.array([r["successful_steps_per_episode"] for r in iteration_results])

    smoothed_success = np.array(
        [moving_average(success_mat[i], window) for i in range(n_runs)]
    )
    smoothed_holes = np.array(
        [moving_average(holes_mat[i], window) for i in range(n_runs)]
    )
    smoothed_steps = np.array(
        [moving_nanmean(steps_mat[i], window) for i in range(n_runs)]
    )
    cum_holes = np.array([np.cumsum(holes_mat[i]) for i in range(n_runs)])

    x_smooth = np.arange(window, n_episodes + 1)
    x_full = np.arange(1, n_episodes + 1)

    # moyenne et IC95 entre runs
    s_mean, s_ci = mean_ci95(smoothed_success)
    h_mean, h_ci = mean_ci95(smoothed_holes)
    st_mean, st_ci = mean_ci95(smoothed_steps)
    ch_mean, ch_ci = mean_ci95(cum_holes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Stratégie : {strategy}  |  Env : {env}  |  "
        f"{n_episodes} épisodes  |  {n_runs} run(s)  |  fenêtre = {window}",
        fontsize=11,
    )

    # taux de succès
    ax = axes[0, 0]
    ax.plot(x_smooth, s_mean, color="steelblue", linewidth=1.2)
    ax.fill_between(
        x_smooth, s_mean - s_ci, s_mean + s_ci, alpha=0.25, color="steelblue"
    )
    ax.set_title(f"Taux de succès (final : {s_mean[-1]:.1%})")
    ax.set_ylabel("Taux")
    ax.set_ylim(0, 1)

    # taux de chute
    ax = axes[0, 1]
    ax.plot(x_smooth, h_mean, color="firebrick", linewidth=1.2)
    ax.fill_between(
        x_smooth, h_mean - h_ci, h_mean + h_ci, alpha=0.25, color="firebrick"
    )
    ax.set_title(f"Taux de chute (final : {h_mean[-1]:.1%})")
    ax.set_ylabel("Taux")
    ax.set_ylim(0, 1)

    # chutes cumulées
    ax = axes[1, 0]
    ax.plot(x_full, ch_mean, color="darkorange", linewidth=1.2)
    ax.fill_between(
        x_full, ch_mean - ch_ci, ch_mean + ch_ci, alpha=0.25, color="darkorange"
    )
    ax.set_title(f"Chutes cumulées (total : {ch_mean[-1]:.0f})")
    ax.set_ylabel("Nb de chutes")

    # longueur des épisodes réussis
    ax = axes[1, 1]
    ax.plot(x_smooth, st_mean, color="mediumpurple", linewidth=1.2)
    ax.fill_between(
        x_smooth, st_mean - st_ci, st_mean + st_ci, alpha=0.25, color="mediumpurple"
    )
    ax.set_title("Longueur moyenne des épisodes réussis")
    ax.set_ylabel("Steps")

    for ax in axes.flat:
        ax.set_xlabel("Épisode")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {save_path}")
    else:
        plt.show()
    plt.close()


def plot_policy(q_table, desc, title="", save_path=None):
    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    colors = {"S": "#90EE90", "F": "white", "H": "#FFB3B3", "G": "#87CEEB"}

    if isinstance(desc, np.ndarray):
        rows = [
            "".join(c.decode() if isinstance(c, bytes) else c for c in row)
            for row in desc
        ]
    else:
        rows = list(desc)

    n_rows = len(rows)
    n_cols = len(rows[0])

    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    if title:
        ax.set_title(title, fontsize=12)

    for r in range(n_rows):
        for c in range(n_cols):
            cell = rows[r][c]
            state = r * n_cols + c
            color = colors.get(cell, "white")
            ax.add_patch(
                plt.Rectangle(
                    (c, n_rows - 1 - r),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1,
                )
            )
            if cell == "F" and q_table is not None:
                best_action = int(np.argmax(q_table[state]))
                ax.text(
                    c + 0.5,
                    n_rows - 1 - r + 0.5,
                    action_symbols[best_action],
                    ha="center",
                    va="center",
                    fontsize=16,
                )
            else:
                ax.text(
                    c + 0.5,
                    n_rows - 1 - r + 0.5,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
