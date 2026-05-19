##
#
#  Deterministic single-environment G1 rollout under the trained policy.
#  Uses CPU MuJoCo (no warp) and launches the passive viewer for playback.
#
##

import argparse
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

# allow `from policy.config import G1Config` when launched from repo root or this dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.config import G1Config


############################################################################
# POLICY NETWORK
############################################################################

class ActorMLP(nn.Module):
    """RSL-RL actor MLP with observation normalization."""

    def __init__(self, input_size=80, hidden_sizes=(512, 256, 128), output_size=23):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.mlp = nn.Sequential(*layers)

        self.register_buffer("obs_mean", torch.zeros(1, input_size))
        self.register_buffer("obs_var", torch.ones(1, input_size))

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-2)
        return self.mlp(obs_norm)


def load_policy(checkpoint_path: str, device: torch.device) -> ActorMLP:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_sd = ckpt["actor_state_dict"]

    policy = ActorMLP().to(device)
    weight_map = {}
    for k, v in actor_sd.items():
        if k.startswith("mlp."):
            weight_map[k] = v
        elif k == "obs_normalizer._mean":
            weight_map["obs_mean"] = v
        elif k == "obs_normalizer._var":
            weight_map["obs_var"] = v
    policy.load_state_dict(weight_map, strict=False)
    policy.eval()
    return policy


############################################################################
# OBSERVATION / CONTROL
############################################################################

def gravity_from_quat(quat: np.ndarray) -> np.ndarray:
    """Projected gravity in body frame from base quaternion (w,x,y,z)."""
    qw, qx, qy, qz = quat
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return np.array([gx, gy, gz], dtype=np.float32)


def build_observation(mj_data, robot_cfg, prev_action, nu, device):
    """Build the 80-dim observation vector (matches parallel_sim.py layout)."""
    omega = mj_data.sensordata[14:17].astype(np.float32)
    quat = mj_data.qpos[3:7].astype(np.float32)
    gravity = gravity_from_quat(quat)
    qpos_j = mj_data.qpos[7:7 + nu].astype(np.float32)
    qvel_j = mj_data.qvel[6:6 + nu].astype(np.float32)

    cmd = np.asarray(robot_cfg.cmd, dtype=np.float32)
    default_j = np.asarray(robot_cfg.default_joint_pos, dtype=np.float32)

    phase = (mj_data.time % robot_cfg.gait_period) / robot_cfg.gait_period
    gait_phase = np.array(
        [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
        dtype=np.float32,
    )
    if np.linalg.norm(cmd) < robot_cfg.stand_cmd_threshold:
        gait_phase[:] = 0.0

    obs = np.concatenate([
        omega,                  # 0:3
        gravity,                # 3:6
        cmd,                    # 6:9
        gait_phase,             # 9:11
        qpos_j - default_j,     # 11:34
        qvel_j,                 # 34:57
        prev_action,            # 57:80
    ]).astype(np.float32)

    return torch.from_numpy(obs).unsqueeze(0).to(device)


def compute_torque(action_np, mj_data, robot_cfg, nu):
    """PD torque from raw policy output."""
    qpos_des = action_np * robot_cfg.action_scale + robot_cfg.default_joint_pos
    qpos_j = mj_data.qpos[7:7 + nu]
    qvel_j = mj_data.qvel[6:6 + nu]
    return robot_cfg.Kp * (qpos_des - qpos_j) + robot_cfg.Kd * (-qvel_j)


############################################################################
# MAIN
############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="models/g1_23dof.xml",
                        help="path to MuJoCo XML")
    parser.add_argument("--policy", default="policy/g1_23dof_vel.pt",
                        help="path to RSL-RL .pt checkpoint")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="rollout length in seconds")
    parser.add_argument("--sim_dt", type=float, default=0.001,
                        help="physics timestep")
    parser.add_argument("--headless", action="store_true",
                        help="run without the viewer (no real-time pacing)")
    args = parser.parse_args()

    # determinism
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(False)

    device = torch.device("cpu")

    # robot + model setup
    robot_cfg = G1Config()
    nu = len(robot_cfg.Kp)

    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_model.opt.timestep = args.sim_dt
    mj_data = mujoco.MjData(mj_model)

    control_ratio = robot_cfg.control_dt / args.sim_dt
    if not math.isclose(control_ratio, round(control_ratio), abs_tol=1e-9):
        raise ValueError(
            f"sim_dt ({args.sim_dt}) must divide control_dt "
            f"({robot_cfg.control_dt}) exactly."
        )
    decimation = int(round(control_ratio))

    # deterministic initial condition = nominal pose, zero velocity
    mj_data.qpos[:7] = robot_cfg.default_base_pos
    mj_data.qpos[7:7 + nu] = robot_cfg.default_joint_pos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # policy
    policy = load_policy(args.policy, device)
    print(f"Loaded policy from [{args.policy}].")
    print(f"sim_dt={args.sim_dt}, control_dt={robot_cfg.control_dt}, "
          f"decimation={decimation}, duration={args.duration}s")

    n_steps = int(round(args.duration / args.sim_dt))
    prev_action = np.zeros(nu, dtype=np.float32)
    action = prev_action.copy()

    def step_once(step):
        nonlocal prev_action, action
        if step % decimation == 0:
            obs = build_observation(mj_data, robot_cfg, prev_action, nu, device)
            action = policy(obs).squeeze(0).cpu().numpy().astype(np.float32)
            prev_action = action
        mj_data.ctrl[:] = compute_torque(action, mj_data, robot_cfg, nu)
        mujoco.mj_step(mj_model, mj_data)

    if args.headless:
        t0 = time.time()
        for step in range(n_steps):
            step_once(step)
        print(f"Headless rollout done in {time.time() - t0:.2f}s "
              f"({n_steps} steps).")
        return

    # interactive viewer with real-time pacing
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        real_t0 = time.time()
        for step in range(n_steps):
            if not viewer.is_running():
                break
            step_once(step)
            viewer.sync()
            # real-time pacing
            sim_elapsed = (step + 1) * args.sim_dt
            real_elapsed = time.time() - real_t0
            if sim_elapsed > real_elapsed:
                time.sleep(sim_elapsed - real_elapsed)


if __name__ == "__main__":
    main()
