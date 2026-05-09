import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import mujoco
import random
import time

def set_110_degree(env):
    # for 110 degree
    model = env.unwrapped.model
    data = env.unwrapped.data
    def jnt(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    def act(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    # door_hinge
    jnt_id = jnt("door_hinge")
    model.jnt_range[jnt_id] = np.array([0., np.pi])
    # ARTz
    ARTZ_id = jnt("ARTz")
    model.jnt_range[ARTZ_id] = np.array([-1., 1.5])
    ARTZ_act_id = act("A_ARTz")
    model.actuator_ctrlrange[ARTZ_act_id] = np.array([-1., 1.5])
    # ARRx
    ARRX_id = jnt("ARRx")
    model.jnt_range[ARRX_id] = np.array([-1.5, 1.5])
    ARRX_act_id = act("A_ARRx")
    model.actuator_ctrlrange[ARRX_act_id] = np.array([-1.5, 1.5])
    # ARRy
    ARRY_id = jnt("ARRy")
    model.jnt_range[ARRY_id] = np.array([-1.5, 1.5])
    ARRY_act_id = act("A_ARRY")
    model.actuator_ctrlrange[ARRY_act_id] = np.array([-1.5, 1.5])
    # Forward 更新
    mujoco.mj_forward(model, data)

def set_close(env):
    # for close
    qs_cur = env.unwrapped.get_env_state()
    qs_cur["qpos"][27] = np.pi * 0
    qs_cur["qpos"][28] = np.pi / 4
    qs_cur["qpos"][29] = np.pi / 2 * 0.9
    qs_cur["qpos"][0] -= 0.35
    env.unwrapped.set_env_state(qs_cur)
    o = env.unwrapped._get_obs()
    env.unwrapped.set_env_state(qs_cur)
    return o

def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat
    
def set_pen_orientation(env):
    # desired_orien = np.zeros(3)
    # desired_orien[0] = np.random.uniform(low=-1, high=1)
    # desired_orien[1] = np.random.uniform(low=-0.5, high=0)  # if 1.57 then [1,0,0]
    desired_orien = np.array([-0.6849743 , -0.83325798, 0.], dtype=np.float64)
    qs_cur = env.unwrapped.get_env_state()
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, desired_orien, 'xyz')
    qs_cur['desired_orien'] = quat
    env.unwrapped.set_env_state(qs_cur)
    o = env.unwrapped._get_obs()
    env.unwrapped.set_env_state(qs_cur)
    return o, desired_orien


def set_relocate_target(env):
    target = np.array([-0.00644676, 0.00177799,  0.2729162], dtype=np.float64)
    object_pos = np.array([-0.11624326, 0.27429217, 0.035], dtype=np.float64)
    qs_cur = env.unwrapped.get_env_state()
    qs_cur = {
        'qpos': qs_cur['qpos'].copy(),
        'qvel': qs_cur['qvel'].copy(),
        'obj_pos': object_pos.copy(),  # obj_pos': qs_cur['obj_pos'].copy()
        'target_pos': target.copy(),
    }

    env.unwrapped.set_env_state(qs_cur)
    o = env.unwrapped._get_obs()
    env.unwrapped.set_env_state(qs_cur)
    return o


def quat_angle_error_acos(pen_roat, target_roat, eps=1e-6):
    """
    q1, q2: (..., 4), wxyz or xyzw 必须一致
    return: (...,) rotation angle in [0, pi], 0表示方向一致
    """
    q1, q2 = np.zeros(4), np.zeros(4)
    mujoco.mju_euler2Quat(q1, pen_roat, 'xyz')
    mujoco.mju_euler2Quat(q2, target_roat, 'xyz')
    q1, q2 = torch.as_tensor(q1, dtype=torch.float32), torch.as_tensor(q2, dtype=torch.float32)
    # 单位化
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + eps)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + eps)

    # 内积
    dot = torch.sum(q1 * q2, dim=-1)

    # # 数值稳定：避免 abs
    # dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    # # 相对旋转角
    # angle = 2.0 * torch.acos(dot.abs())

    # dot ∈ [0, 1] ，dot越大方向越接近
    return dot

def quat_distance_sin2(pen_roat, target_roat, eps=1e-6):
    q1, q2 = np.zeros(4), np.zeros(4)
    mujoco.mju_euler2Quat(q1, pen_roat, 'xyz')
    mujoco.mju_euler2Quat(q2, target_roat, 'xyz')
    q1, q2 = torch.as_tensor(q1, dtype=torch.float32), torch.as_tensor(q2, dtype=torch.float32)
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + eps)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + eps)

    dot = torch.sum(q1 * q2, dim=-1)
    dist = 1.0 - dot**2
    return dist

def quat_angle_error_acos_batch(pen_rot, target_rot, eps=1e-6):
    """
    pen_rot, target_rot:
        shape (3,) or (B, 3), Euler angles in xyz order (rad)

    return:
        angle: shape (B,), rotation error in [0, pi]
    """
    pen_rot = np.asarray(pen_rot)
    target_rot = np.asarray(target_rot)

    if pen_rot.ndim == 1:
        pen_rot = pen_rot[None, :]

    B = pen_rot.shape[0]
    q1 = np.zeros((B, 4))
    q2 = np.zeros((1, 4))
    mujoco.mju_euler2Quat(q2[0], target_rot, 'xyz')
    q2 = np.repeat(q2, B, axis=0)

    for i in range(B):
        mujoco.mju_euler2Quat(q1[i], pen_rot[i], 'xyz')

    q1 = torch.from_numpy(q1)
    q2 = torch.from_numpy(q2)

    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + eps)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + eps)
    dot = torch.sum(q1 * q2, dim=-1)
    dot = torch.clamp(torch.abs(dot), -1.0 + eps, 1.0 - eps)
    angle = 2.0 * torch.acos(dot)
    return angle

def quat_distance_sin2_batch(pen_rot, target_rot, eps=1e-6):
    """
    pen_rot, target_rot:
        shape (3,) or (B, 3), Euler angles (xyz), rad

    return:
        dist: shape (B,), sin^2(theta / 2) in [0, 1]
              0 表示方向一致
    """
    pen_rot = np.asarray(pen_rot)
    target_rot = np.asarray(target_rot)

    if pen_rot.ndim == 1:
        pen_rot = pen_rot[None, :]
    if target_rot.ndim == 1:
        target_rot = target_rot[None, :]

    B = pen_rot.shape[0]

    q1 = np.zeros((B, 4))
    q2 = np.zeros((1, 4))
    mujoco.mju_euler2Quat(q2[0], target_rot[0], 'xyz')
    q2 = np.repeat(q2, B, axis=0)

    for i in range(B):
        mujoco.mju_euler2Quat(q1[i], pen_rot[i], 'xyz')

    q1 = torch.from_numpy(q1)
    q2 = torch.from_numpy(q2)
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + eps)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + eps)

    dot = torch.sum(q1 * q2, dim=-1)
    dist = 1.0 - dot ** 2
    return dist

class ExpertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {
            "obs": np.array(self.dataset[idx].observations, dtype=np.float32),
            "acts": np.array(self.dataset[idx].actions, dtype=np.float32),
        }
        return sample
    
class ExpertDatasetPen1(Dataset):
    def __init__(self, dataset, idx=None):
        self.idx = idx
        if idx:
            self.dataset = []
            for i in idx:
                self.dataset.append(dataset[i].observations)
        else:
            self.dataset = dataset
        self.dataset_s = dataset

    def __len__(self):
        if self.idx:
            return len(self.idx)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            sample_obs = np.concatenate([self.dataset[i].observations.reshape(-1, 45) for i in idx], axis=0)
            sample_acts = np.concatenate([self.dataset[i].actions.reshape(-1, 24) for i in idx], axis=0)
        else:
            i = self.idx[idx]
            sample_obs = np.array(self.dataset_s[i].observations.reshape(-1, 45), dtype=np.float32)
        return sample_obs
    
class ExpertDatasetPen2(Dataset):
    def __init__(self, dataset, idx=None):
        self.idx = idx
        self.dataset = dataset
        if idx:
            self.dataset_s = []
            self.dataset_a = []
            for i in idx:
                self.dataset_s.append(dataset[i].observations)
                self.dataset_a.append(dataset[i].actions)
        else:
            self.dataset = dataset

    def __len__(self):
        if self.idx:
            return len(self.idx)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            sample_obs = np.concatenate([self.dataset[i].observations.reshape(-1, 45) for i in idx], axis=0)
            sample_acts = np.concatenate([self.dataset[i].actions.reshape(-1, 24) for i in idx], axis=0)
        else:
            i = self.idx[idx]
            sample_obs = np.array(self.dataset[i].observations[:-1].reshape(-1, 45), dtype=np.float32)
            sample_acts = np.array(self.dataset[i].actions.reshape(-1, 24), dtype=np.float32)
        return sample_obs, sample_acts
    
class ExpertDatasetPen(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            sample_obs = np.concatenate([self.dataset[i].observations[:-1].reshape(-1, 45) for i in idx], axis=0)
            sample_acts = np.concatenate([self.dataset[i].actions[:-1].reshape(-1, 24) for i in idx], axis=0)
        else:
            sample_obs = np.array(self.dataset[idx].observations[:-1].reshape(-1, 45), dtype=np.float32)
            sample_acts = np.array(self.dataset[idx].actions[:-1].reshape(-1, 24), dtype=np.float32)
        return sample_obs, sample_acts
    
class ExpertDatasetRelocate(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.obs_shape = self.dataset[0].observations.shape[-1]
        self.act_shape = self.dataset[0].actions.shape[-1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            sample_obs = np.concatenate([self.dataset[i].observations[:-1].reshape(-1, self.obs_shape) for i in idx], axis=0)
            sample_acts = np.concatenate([self.dataset[i].actions.reshape(-1, self.act_shape) for i in idx], axis=0)
        else:
            sample_obs = np.array(self.dataset[idx].observations[:-1].reshape(-1, self.obs_shape), dtype=np.float32)
            sample_acts = np.array(self.dataset[idx].actions.reshape(-1, self.act_shape), dtype=np.float32)
        return sample_obs, sample_acts
    
class ExpertDatasetRelocateNEAR(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.obs_shape = self.dataset[0].observations.shape[-1]
        self.act_shape = self.dataset[0].actions.shape[-1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            sample_obs = np.stack([self.dataset[i].observations for i in idx], axis=0)
            sample_acts = np.stack([self.dataset[i].actions for i in idx], axis=0)
        else:
            sample_obs = np.array(self.dataset[idx].observations.reshape(-1, self.obs_shape), dtype=np.float32)
            sample_acts = np.array(self.dataset[idx].actions.reshape(-1, self.act_shape), dtype=np.float32)
        return sample_obs, sample_acts
    

def sample_batch_qpos(dataset, batch_size):
    dataset_obs = dataset.fields.observations
    dataset_act = dataset.fields.actions
    N = len(dataset_act)
    # 随机索引 batch_size 个样本
    indices = torch.randint(low=0, high=N, size=(batch_size,))
    batch_obs = []
    batch_acts = []
    for i in range(batch_size):
        obs = dataset_obs[indices[i]]
        acts = dataset_act[indices[i]]
        batch_obs.append(obs)
        batch_acts.append(acts)
    batch_obs = np.array(batch_obs)
    batch_acts = np.array(batch_acts)
    
    return batch_obs, batch_acts


def sample_batch(dataset, batch_size):
    N = len(dataset)
    # 随机索引 batch_size 个样本
    indices = torch.randint(low=0, high=N, size=(batch_size,))
    batch_obs = []
    batch_acts = []
    for i in range(batch_size):
        obs = dataset[indices[i]].observations
        acts = dataset[indices[i]].actions
        batch_obs.append(obs)
        batch_acts.append(acts)
    batch_obs = np.array(batch_obs)
    batch_acts = np.array(batch_acts)
    
    return batch_obs, batch_acts

def sample_batch_withR(dataset, batch_size):
    N = len(dataset)
    # 随机索引 batch_size 个样本
    indices = torch.randint(low=0, high=N, size=(batch_size,))
    batch_obs = []
    batch_acts = []
    batch_rewards = []
    for i in range(batch_size):
        obs = dataset[indices[i]].observations
        acts = dataset[indices[i]].actions
        r = dataset[indices[i]].rewards
        batch_obs.append(obs)
        batch_acts.append(acts)
        batch_rewards.append(r)
    batch_obs = np.array(batch_obs)
    batch_acts = np.array(batch_acts)
    batch_rewards = np.array(batch_rewards)
    
    return batch_obs, batch_acts, batch_rewards

def sample_batch_pen(dataset, batch_size, idx=None):
    if idx:
        indices = random.sample(idx, batch_size)
    else:
        N = len(dataset)
        indices = torch.randint(low=0, high=N, size=(batch_size,))
    batch_obs = []
    batch_acts = []
    for i in range(batch_size):
        obs = dataset[indices[i]].observations
        acts = dataset[indices[i]].actions
        batch_obs.append(obs)
        batch_acts.append(acts)
    # batch_obs = np.array(batch_obs)
    # batch_acts = np.array(batch_acts)
    return batch_obs, batch_acts

def sample_batch_array(dataset, batch_size, idx=None):
    # current_seed = int(time.time() % 1000)
    # random.seed(current_seed)
    if idx:
        indices = random.sample(idx, batch_size)
    else:
        N = len(dataset)
        indices = torch.randint(low=0, high=N, size=(batch_size,))
    batch_obs = [] # 2053, 2903, 3482, 3646, 1192 (5 traj no random) # 2053, 2903, 3482, 3646, 1192, 4096, 4957, 440, 2107, 2215 (10 traj no random)
    batch_acts = []

    for i in range(batch_size):
        obs = dataset[indices[i]].observations
        acts = dataset[indices[i]].actions
        batch_obs.append(obs)
        batch_acts.append(acts)
    batch_obs = np.array(batch_obs)
    batch_acts = np.array(batch_acts)
    # random.seed(34)
    return batch_obs, batch_acts

def sample_batch_array_R(dataset, batch_size, idx=None):
    # current_seed = int(time.time() % 1000)
    # random.seed(current_seed)
    if idx:
        indices = random.sample(idx, batch_size)
    else:
        N = len(dataset)
        indices = torch.randint(low=0, high=N, size=(batch_size,))
    batch_obs = [] # 2053, 2903, 3482, 3646, 1192 (5 traj no random) # 2053, 2903, 3482, 3646, 1192, 4096, 4957, 440, 2107, 2215 (10 traj no random)
    batch_acts = []
    batch_rewards = []
    for i in range(batch_size):
        obs = dataset[indices[i]].observations
        acts = dataset[indices[i]].actions
        r = dataset[indices[i]].rewards
        batch_obs.append(obs)
        batch_acts.append(acts)
        batch_rewards.append(r)
    batch_obs = np.array(batch_obs)
    batch_acts = np.array(batch_acts)
    batch_rewards = np.array(batch_rewards)
    # random.seed(34)
    return batch_obs, batch_acts, batch_rewards

def sample_batch_onestep(dataset, batch_size, device="cpu"):
    """
    加速版随机采样专家数据(batch of single-step transitions)
    dataset[i].observations/actions shape: (n_episodes, T, obs_dim / act_dim)
    """
    n_trajs = len(dataset)

    # 1. 随机选 batch_size 条 trajectory
    traj_indices = np.random.randint(0, n_trajs, size=batch_size)

    # 2. 为每条 trajectory 随机选 episode 和 timestep
    ep_indices = []
    t_indices = []
    for idx in traj_indices:
        n_eps, T = dataset[idx].observations.shape[:2]
        ep_indices.append(np.random.randint(0, n_eps-1))
        # t_indices.append(np.random.randint(0, T))

    # 3. 批量采样
    batch_obs = np.array([dataset[i].observations[ep_idx]
                          for i, ep_idx in zip(traj_indices, ep_indices)])
    batch_acts = np.array([dataset[i].actions[ep_idx]
                           for i, ep_idx in zip(traj_indices, ep_indices)])

    # 4. 转 torch tensor
    batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=device)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float32, device=device)

    return batch_obs, batch_acts


def get_init_state(env, data, max_steps_per_episode=20, seed=None):
    env = env.unwrapped

    with torch.no_grad():
        # ep_seed = rng.randint(0, 2**32-1)
        obs, info = env.reset(seed=seed)
        done = False
        step = 0
        ep_reward = 0.0
        weights_data = []
        while not done and step < max_steps_per_episode:
            # 这里用随机策略作示例；替换为你的 policy.sample(obs) 即可
            action = data[step]
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            step += 1

    state = env.get_env_state()
    env.close()
    return state

class LossLogger:
    def __init__(self):
        self.q1_mean = []
        self.q_targ = []
        self.backup = []
        self.log_pi = []
        self.rew_mean = []
        self.q_grad = []
        self.pi_grad = []
        self.td_pos_frac = []
        self.ema = []

    def log(self, q1_mean, q_targ, backup, log_pi, rew_mean, q_grad, pi_grad, td_pos_frac):
        self.q1_mean.append(q1_mean)
        self.q_targ.append(q_targ)
        self.backup.append(backup)
        self.log_pi.append(log_pi)
        self.rew_mean.append(rew_mean)
        self.q_grad.append(q_grad)
        self.pi_grad.append(pi_grad)
        self.td_pos_frac.append(td_pos_frac)

    def save_to_file(self, path):
        import numpy as np
        np.savez(
            path,
            q1_mean=np.array(self.q1_mean),
            q_targ=np.array(self.q_targ),
            backup=np.array(self.backup),
            log_pi=np.array(self.log_pi),
            rew_mean=np.array(self.rew_mean),
            q_grad=np.array(self.q_grad),
            pi_grad=np.array(self.pi_grad),
            td_pos_frac=np.array(self.td_pos_frac),
        )

    def reset(self):
        """清除当前缓存，用于每轮训练结束后重新统计"""
        self.q1_mean.clear()
        self.q_targ.clear()
        self.backup.clear()
        self.log_pi.clear()
        self.rew_mean.clear()
        self.q_grad.clear()
        self.pi_grad.clear()
        self.td_pos_frac.clear()


class RewardNormalizer:
    def __init__(self, eps=1e-8, clip_value=None):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.eps = eps
        self.clip_value = clip_value  # 可选
    
    def update(self, rewards):
        # rewards: torch tensor [batch_size]
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()
        batch_count = rewards.numel()

        new_count = self.count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / new_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / new_count

        new_var = M2 / new_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, rewards):
        rewards_norm = (rewards - self.mean) / (np.sqrt(self.var) + self.eps)
        if self.clip_value is not None:
            rewards_norm = torch.clamp(rewards_norm, -self.clip_value, self.clip_value)
        return rewards_norm

