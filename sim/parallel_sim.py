##
#
#  Parallel G1 simulation using MuJoCo Warp (GPU) + PyTorch policy
#
##

# standard imports
import os
import sys
import time
import math
import numpy as np
from dataclasses import dataclass

# torch imports
import torch

# mujoco, warp imports
import mujoco
import mujoco_warp as mjwarp
import warp as wp

# custom imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.actor import load_policy
from policy.config import G1Config
from utils.math_utils import get_gravity_orientation_batch


############################################################################
# PARALLEL SIM CONFIG
############################################################################

@dataclass
class ParallelSimConfig:

    # simulation parameters
    batch_size: int                            # number of parallel environments
    seed: int                                  # random seed
    
    # model and policy paths
    xml_path: str                              # path to MuJoCo XML
    policy_path: str                           # path to .pt checkpoint
    sim_dt_des: float = None                   # desired sim timestep (None = use XML default)
    robot_config: G1Config = None              # robot config (Kp, Kd, etc.)
    
    # data type to log
    log_dtype: torch.dtype = torch.float32     # data type for logging

    # initial state bounds (if None, defaults to robot config default pose and zero velocity)
    q_lb: np.ndarray = None                    # lower bound on initial qpos, shape (nq,)
    q_ub: np.ndarray = None                    # upper bound on initial qpos, shape (nq,)
    v_lb: np.ndarray = None                    # lower bound on initial qvel, shape (nv,)
    v_ub: np.ndarray = None                    # upper bound on initial qvel, shape (nv,)


############################################################################
# PARALLEL SIM CLASS
############################################################################

class ParallelSim:
    """Parallel G1 rollout using MuJoCo Warp + PyTorch policy."""

    def __init__(self, config: ParallelSimConfig):

        # use GPU for simulation and policy inference
        self.device = torch.device("cuda:0")

        # config parameters
        self.batch_size = config.batch_size
        self.log_dtype = config.log_dtype

        # random seed
        seed = config.seed
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        print(f"Random seed: {seed}")

        # robot config
        self.robot_cfg = config.robot_config or G1Config()
        self.nu = len(self.robot_cfg.Kp)  # 23

        # PD gains and default positions on GPU
        self.Kp = torch.tensor(self.robot_cfg.Kp, dtype=torch.float32, device=self.device)
        self.Kd = torch.tensor(self.robot_cfg.Kd, dtype=torch.float32, device=self.device)
        self.default_joint_pos = torch.tensor(
            self.robot_cfg.default_joint_pos, dtype=torch.float32, device=self.device
        )
        self.action_scale = torch.tensor(
            self.robot_cfg.action_scale, dtype=torch.float32, device=self.device
        )
        self.cmd = torch.tensor(
            self.robot_cfg.cmd, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.batch_size, -1)  # (N, 3)

        # timing
        self.control_dt = self.robot_cfg.control_dt
        self.gait_period = self.robot_cfg.gait_period

        # load mujoco model and data
        self.mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # override sim timestep if desired (this is the timestep used for physics integration and logging)
        if config.sim_dt_des is not None:
            self.mj_model.opt.timestep = config.sim_dt_des
        self.sim_dt = float(self.mj_model.opt.timestep)

        # compute control decimation (number of sim steps per control step)
        control_ratio = self.control_dt / self.sim_dt
        if not math.isclose(control_ratio, round(control_ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"sim_dt ({self.sim_dt}) must divide control_dt ({self.control_dt}) exactly."
            )
        self.decimation = int(round(control_ratio))

        # robot state size
        self.nq = self.mj_model.nq  # 30  (7 base + 23 joints)
        self.nv = self.mj_model.nv  # 29  (6 base + 23 joints)

        # initial condition state bounds
        default_q = np.zeros(self.nq, dtype=np.float32)
        default_q[:7] = self.robot_cfg.default_base_pos
        default_q[7:7 + self.nu] = self.robot_cfg.default_joint_pos
        default_v = np.zeros(self.nv, dtype=np.float32)
        q_lb = default_q if config.q_lb is None else np.asarray(config.q_lb, dtype=np.float32)
        q_ub = default_q if config.q_ub is None else np.asarray(config.q_ub, dtype=np.float32)
        v_lb = default_v if config.v_lb is None else np.asarray(config.v_lb, dtype=np.float32)
        v_ub = default_v if config.v_ub is None else np.asarray(config.v_ub, dtype=np.float32)

        # ensure that bounds have correct shape
        if q_lb.shape != (self.nq,) or q_ub.shape != (self.nq,):
            raise ValueError(f"q_lb and q_ub must each have shape ({self.nq},)")
        if v_lb.shape != (self.nv,) or v_ub.shape != (self.nv,):
            raise ValueError(f"v_lb and v_ub must each have shape ({self.nv},)")
        
        self.q_lb = torch.tensor(q_lb, dtype=torch.float32, device=self.device)
        self.q_ub = torch.tensor(q_ub, dtype=torch.float32, device=self.device)
        self.v_lb = torch.tensor(v_lb, dtype=torch.float32, device=self.device)
        self.v_ub = torch.tensor(v_ub, dtype=torch.float32, device=self.device)

        # set default initial state
        default_base = self.robot_cfg.default_base_pos
        default_joints = self.robot_cfg.default_joint_pos
        self.mj_data.qpos[:7] = default_base
        self.mj_data.qpos[7:7 + self.nu] = default_joints

        # put model and data on GPU
        with wp.ScopedDevice("cuda:0"):
            self.wp_model = mjwarp.put_model(self.mj_model)
            self.wp_data = mjwarp.put_data(
                self.mj_model, self.mj_data, nworld=self.batch_size,
                njmax=200, nconmax=200,
            )

        # zero-copy torch views into warp arrays
        self.t_qpos = wp.to_torch(self.wp_data.qpos)        # (N, nq)
        self.t_qvel = wp.to_torch(self.wp_data.qvel)        # (N, nv)
        self.t_ctrl = wp.to_torch(self.wp_data.ctrl)        # (N, nu)
        self.t_sensordata = wp.to_torch(self.wp_data.sensordata)  # (N, nsensordata)
        self.t_time = wp.to_torch(self.wp_data.time)        # (N,)

        print(f"MuJoCo Warp initialized: nq={self.nq}, nv={self.nv}, nu={self.nu}")
        print(f"  sim_dt=[{self.sim_dt} sec]")
        print(f"  control_dt=[{self.control_dt} sec]")
        print(f"  control decimation=[{self.decimation}]")
        print(f"  batch_size=[{self.batch_size}]")

        # load policy
        self.policy = load_policy(config.policy_path, self.device)
        print(f"Loaded policy from [{config.policy_path}].")

        # pre-allocate observation buffer
        self.obs_size = 80
        self.obs_buf = torch.zeros(self.batch_size, self.obs_size,
                                   dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #

    def build_observation(self, prev_action: torch.Tensor, sim_time: torch.Tensor) -> torch.Tensor:
        """
        Build the 80-dim observation vector from simulation state.

        Args:
            prev_action: (N, 23) previous policy output
            sim_time:    (N,)    simulation time per env
        Returns:
            obs: (N, 80)
        """
        # from sensordata: pelvis gyro (body-frame angular velocity)
        # sensor layout: torso_quat(4), torso_gyro(3), torso_acc(3),  [0:10]
        #                pelvis_quat(4), pelvis_gyro(3), pelvis_acc(3) [10:20]
        #                left_foot_touch(1), right_foot_touch(1)       [20:22]
        omega_body = self.t_sensordata[:, 14:17]       # (N, 3)

        # base quaternion for gravity projection
        quat = self.t_qpos[:, 3:7]                     # (N, 4)
        gravity = get_gravity_orientation_batch(quat)   # (N, 3)

        # joint state
        qpos_joints = self.t_qpos[:, 7:7 + self.nu]   # (N, 23)
        qvel_joints = self.t_qvel[:, 6:6 + self.nu]   # (N, 23)

        # gait phase clock
        phase = (sim_time % self.gait_period) / self.gait_period  # (N,)
        two_pi = 2.0 * math.pi
        gait_sin = torch.sin(two_pi * phase)   # (N,)
        gait_cos = torch.cos(two_pi * phase)   # (N,)
        gait_phase = torch.stack([gait_sin, gait_cos], dim=-1)  # (N, 2)

        # zero gait phase when command is below threshold
        cmd_norm = torch.norm(self.cmd, dim=-1)  # (N,)
        standing = (cmd_norm < self.robot_cfg.stand_cmd_threshold)
        gait_phase[standing] = 0.0

        # joint position error
        qj_err = qpos_joints - self.default_joint_pos  # (N, 23)

        # assemble observation:
        # [omega(3), gravity(3), cmd(3), phase(2), qj_err(23), dqj(23), prev_action(23)]
        obs = torch.cat([
            omega_body,                      # 0:3
            gravity,                         # 3:6
            self.cmd,                        # 6:9
            gait_phase,                      # 9:11
            qj_err,                          # 11:34
            qvel_joints,                     # 34:57
            prev_action,                     # 57:80
        ], dim=-1)

        return obs

    # ------------------------------------------------------------------ #
    #  PD Control
    # ------------------------------------------------------------------ #

    def compute_torque(self, action: torch.Tensor) -> torch.Tensor:
        """
        PD control: action → desired position → torque.

        Args:
            action: (N, 23) raw policy output
        Returns:
            tau: (N, 23)
        """
        qpos_des = action * self.action_scale + self.default_joint_pos
        qpos_joints = self.t_qpos[:, 7:7 + self.nu]
        qvel_joints = self.t_qvel[:, 6:6 + self.nu]
        tau = self.Kp * (qpos_des - qpos_joints) + self.Kd * (0.0 - qvel_joints)
        return tau

    # ------------------------------------------------------------------ #
    #  Initial Conditions
    # ------------------------------------------------------------------ #

    def sample_random_uniform_initial_conditions(self):
        """
        Sample initial conditions uniformly from configured q/v bounds.

        Returns:
            q0_batch: (N, nq) initial positions
            v0_batch: (N, nv) initial velocities
        """
        N = self.batch_size

        q_rand = torch.rand(N, self.nq, dtype=torch.float32, device=self.device, generator=self.rng)
        v_rand = torch.rand(N, self.nv, dtype=torch.float32, device=self.device, generator=self.rng)

        q0_batch = self.q_lb.unsqueeze(0) + (self.q_ub - self.q_lb).unsqueeze(0) * q_rand
        v0_batch = self.v_lb.unsqueeze(0) + (self.v_ub - self.v_lb).unsqueeze(0) * v_rand

        return q0_batch, v0_batch

    # ------------------------------------------------------------------ #
    #  Rollout
    # ------------------------------------------------------------------ #

    def rollout_policy_input(self, T: int):
        """
        Sample random ICs and run a policy rollout.

        Args:
            T: number of sim steps (T-1 integrations, logged at sim_dt)
        Returns:
            q_log: (N, T, nq) joint positions
            v_log: (N, T, nv) joint velocities
        """
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()
        return self._rollout_policy_input(q0_batch, v0_batch, T)

    def _rollout_policy_input(self, q0_batch: torch.Tensor, v0_batch: torch.Tensor, T: int):
        """Run T simulation steps with the policy from given ICs.

        Args:
            q0_batch: (N, nq) initial positions
            v0_batch: (N, nv) initial velocities
            T:        number of sim steps (T-1 integrations)
        Returns:
            q_log: (N, T, nq) joint positions  (logged at sim_dt)
            v_log: (N, T, nv) joint velocities (logged at sim_dt)
        """
        N = self.batch_size
        S = T - 1  # number of steps to take (T includes initial state)

        # reset data to default state
        mjwarp.reset_data(self.wp_model, self.wp_data)

        # set initial conditions from inputs
        self.t_qpos[:] = q0_batch
        self.t_qvel[:] = v0_batch

        # run forward to initialize sensors/kinematics
        with wp.ScopedDevice("cuda:0"):
            mjwarp.forward(self.wp_model, self.wp_data)

        # logging at sim_dt
        q_log = torch.zeros(N, T, self.nq, dtype=self.log_dtype, device=self.device)
        v_log = torch.zeros(N, T, self.nv, dtype=self.log_dtype, device=self.device)

        # log initial state
        q_log[:, 0, :] = self.t_qpos.to(self.log_dtype)
        v_log[:, 0, :] = self.t_qvel.to(self.log_dtype)

        # initialize previous action
        prev_action = torch.zeros(N, self.nu, dtype=torch.float32, device=self.device)
        action = prev_action

        # main rollout loop
        with wp.ScopedDevice("cuda:0"):
            for step in range(S):
                # update policy every decimation steps
                if step % self.decimation == 0:
                    sim_time = self.t_time  # (N,)
                    obs = self.build_observation(prev_action, sim_time)
                    action = self.policy(obs)  # (N, 23)
                    prev_action = action

                # PD torque and physics step
                tau = self.compute_torque(action)
                self.t_ctrl[:] = tau
                mjwarp.step(self.wp_model, self.wp_data)

                # log state
                q_log[:, step + 1, :] = self.t_qpos.to(self.log_dtype)
                v_log[:, step + 1, :] = self.t_qvel.to(self.log_dtype)

        return q_log, v_log

    def rollout_policy_input_x0(self, x0_batch: torch.Tensor, T: int):
        """Run a policy rollout from a stacked state vector x0 = [q, v].

        Args:
            x0_batch: (N, nq+nv) initial states
            T:        number of sim steps (T-1 integrations)
        Returns:
            q_log: (N, T, nq) joint positions
            v_log: (N, T, nv) joint velocities
        """
        q0_batch = x0_batch[:, :self.nq]
        v0_batch = x0_batch[:, self.nq:self.nq + self.nv]
        return self._rollout_policy_input(q0_batch, v0_batch, T)


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # print the device being used
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # paths
    xml_path = "./models/g1_23dof.xml"
    policy_path = "./policy/g1_23dof_vel.pt"

    # environment name (used for file naming)
    env_name = "g1_23dof"

    # rollout parameters
    batch_size = 1024
    sim_dt_des = 0.002  # dt for simulation and logging (too low makes sim instability)
    T = 2500            # number of sim steps to rollout (e.g., 2500 steps at 2ms sim_dt = 5.0 sec)

    # number of datasets to generate
    N_datasets = 10

    # random seed
    # seed = 0
    seed = int(time.time())

    # create the config for parallel sim
    config = ParallelSimConfig(
        batch_size=batch_size,
        xml_path=xml_path,
        policy_path=policy_path,
        seed=seed,
        sim_dt_des=sim_dt_des,
    )

    # create the rollout instance
    r = ParallelSim(config)

    # function to rollout and save a dataset, 
    # returns time it took to complete the rollout (excluding saving time)
    def rollout_and_save_dataset(save_path, label):

        print(f"\nGenerating {label}...")

        t0 = time.time()
        q_log, v_log = r.rollout_policy_input(T)
        torch.cuda.synchronize()
        rollout_time = time.time() - t0
        print(f"Rollout time: {rollout_time:.3f}s")

        q_log_np = q_log.cpu().numpy()
        v_log_np = v_log.cpu().numpy()

        print(f"  q_log: {q_log_np.shape}")
        print(f"  v_log: {v_log_np.shape}")

        np.savez(save_path,
                 sim_dt=float(r.sim_dt),
                 control_dt=float(r.control_dt),
                 q_log=q_log_np,
                 v_log=v_log_np)
        print(f"  Saved to: {save_path}")

        return rollout_time

    # generate numbered datasets: <env_name>_data_XX.npz
    tot_time = 0.0
    save_dir = "./data/data"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(N_datasets):
        save_path = f"{save_dir}/{env_name}_data_{i+1:02d}.npz"
        tot_time += rollout_and_save_dataset(save_path, f"dataset {i+1}/{N_datasets}")

    print(f"\nFinished generating {N_datasets} datasets.")
    print(f"Total rollout time: {tot_time:.1f}s")
    print(f"Total trajectories: {N_datasets * batch_size}")
