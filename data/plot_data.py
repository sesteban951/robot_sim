##
#
#  Plot the full state of randomly sampled trajectories from a parallel_sim .npz dataset.
#
##

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# project root so we can import utils.math_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_utils import body_to_world


############################################################################
# USEFUL VARS AND FUNCTIONS
############################################################################

# state labels — 30 generalized positions, 29 generalized velocities
Q_LABELS = ["px [m]", "py [m]", "pz [m]", "qw", "qx", "qy", "qz"] + [f"q_joint[{i}] [rad]" for i in range(23)]
V_LABELS = ["vx [m/s]", "vy [m/s]", "vz [m/s]", "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"] + [f"v_joint[{i}] [rad/s]" for i in range(23)]


def plot_data(t, X, labels, title, n_cols=5, overlays=None, save_path=None):
    """Plot every state dimension as its own subplot, with one line per trajectory.

    overlays: optional dict {subplot_index -> (num, T) array} drawn as dashed
              lines matching each trajectory's color.
    """

    n_dim = X.shape[2]
    n_rows = (n_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 1.8 * n_rows),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for d in range(n_dim):
        ax = axes[d]
        # X[traj, time, dim] -> line per trajectory
        lines = ax.plot(t, X[:, :, d].T, linewidth=0.8, alpha=0.8)
        # optional overlay: dashed line per trajectory in the matching color
        if overlays is not None and d in overlays:
            ov = overlays[d]  # (num, T)
            for traj_i, line in enumerate(lines):
                ax.plot(t, ov[traj_i], linestyle="--",
                        color=line.get_color(), linewidth=0.8, alpha=0.8)
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
    num_trajectories = 3

    # choose the random seed to select the same trajectories
    # seed = 0
    seed = int(time.time())

    # load the data
    data = np.load(data_path)
    q = data["q_log"]                    # (B, T, nq)
    v = data["v_log"]                    # (B, T, nv)
    cmd = data["cmd_log"]                # (B, T, 3)
    sim_dt = float(data["sim_dt"])
    control_dt = float(data["control_dt"])
    B, T, nq = q.shape
    nv = v.shape[2]
    print(f"loaded: {data_path}")
    print(f"  B={B}, T={T}, nq={nq}, nv={nv}")
    print(f"  sim_dt={sim_dt}, control_dt={control_dt}")

    # sample trajectories to plot
    num = min(num_trajectories, B)
    rng = np.random.default_rng(seed)
    idx = rng.choice(B, size=num, replace=False)
    idx.sort()
    print(f"selected {num} trajectories: {idx.tolist()}")

    # select the lucky trajectories and plot
    q_sel = q[idx]                    # (num, T, nq)
    v_sel = v[idx]                    # (num, T, nv)
    cmd_sel = cmd[idx]                # (num, T, 3)
    t = np.arange(T) * sim_dt         # (T,)

    # cmd is in the body frame; rotate it to world frame to match v_log
    # (which is the free-joint qvel — also world frame).
    quat_sel = q_sel[:, :, 3:7]                                              # (num, T, 4) wxyz
    zero_col = np.zeros_like(cmd_sel[..., 0:1])
    cmd_lin_body = np.concatenate([cmd_sel[..., 0:2], zero_col], axis=-1)    # (num, T, 3) (vx, vy, 0)
    cmd_ang_body = np.concatenate([zero_col, zero_col, cmd_sel[..., 2:3]], axis=-1)  # (num, T, 3) (0, 0, wz)
    cmd_lin_world = body_to_world(quat_sel, cmd_lin_body)                    # (num, T, 3)
    cmd_ang_world = body_to_world(quat_sel, cmd_ang_body)                    # (num, T, 3)

    # overlay world-frame cmd on the matching base-velocity subplots:
    #   V_LABELS[0]=vx → cmd_lin_world[..., 0]
    #   V_LABELS[1]=vy → cmd_lin_world[..., 1]
    #   V_LABELS[5]=wz → cmd_ang_world[..., 2]
    v_overlays = {
        0: cmd_lin_world[:, :, 0],
        1: cmd_lin_world[:, :, 1],
        5: cmd_ang_world[:, :, 2],
    }

    plot_data(t, q_sel, Q_LABELS, f"positions, q")
    plot_data(t, v_sel, V_LABELS, f"velocities, v", overlays=v_overlays)

    plt.show()


if __name__ == "__main__":
    main()
