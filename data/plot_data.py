##
#
#  Plot the full state of randomly sampled trajectories from a parallel_sim .npz dataset.
#
##

import os
import time

import numpy as np
import matplotlib.pyplot as plt

# state labels — 30 generalized positions, 29 generalized velocities
Q_LABELS = ["px [m]", "py [m]", "pz [m]", "qw", "qx", "qy", "qz"] + [f"q_joint[{i}] [rad]" for i in range(23)]
V_LABELS = ["vx [m/s]", "vy [m/s]", "vz [m/s]", "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"] + [f"v_joint[{i}] [rad/s]" for i in range(23)]


def plot_data(t, X, labels, title, save_path=None):
    """Plot every state dimension as its own subplot, with one line per trajectory."""

    n_dim = X.shape[2]
    n_cols = 5
    n_rows = (n_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 1.8 * n_rows),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for d in range(n_dim):
        ax = axes[d]
        # X[traj, time, dim] -> line per trajectory
        ax.plot(t, X[:, :, d].T, linewidth=0.8, alpha=0.8)
        ax.set_title(labels[d], fontsize=9)
        ax.grid(True, alpha=0.3)
        if d // n_cols == n_rows - 1:
            ax.set_xlabel("time [s]", fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for d in range(n_dim, len(axes)):
        axes[d].axis("off")

    fig.suptitle(title, fontsize=11)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")


############################################################################
# MAIN
############################################################################

def main():

    # configuration
    data_path = "./data/data/g1_23dof_data_01.npz"

    # trajectories to plot
    num_trajectories = 10

    # choose the random seed to select the same trajectories
    # seed = 0
    seed = int(time.time())

    # load the data
    data = np.load(data_path)
    q = data["q_log"]                    # (N, T, nq)
    v = data["v_log"]                    # (N, T, nv)
    sim_dt = float(data["sim_dt"])
    control_dt = float(data["control_dt"])
    N, T, nq = q.shape
    nv = v.shape[2]
    print(f"loaded: {data_path}")
    print(f"  N={N}, T={T}, nq={nq}, nv={nv}")
    print(f"  sim_dt={sim_dt}, control_dt={control_dt}")

    # sample trajectories to plot
    num = min(num_trajectories, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=num, replace=False)
    idx.sort()
    print(f"selected {num} trajectories: {idx.tolist()}")

    # select the lucky trajectories and plot
    q_sel = q[idx]                    # (num, T, nq)
    v_sel = v[idx]                    # (num, T, nv)
    t = np.arange(T) * sim_dt         # (T,)

    base = os.path.splitext(os.path.basename(data_path))[0]
    plot_data(t, q_sel, Q_LABELS, f"positions, q")
    plot_data(t, v_sel, V_LABELS, f"velocities, v")

    plt.show()


if __name__ == "__main__":
    main()
