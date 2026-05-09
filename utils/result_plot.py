import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
import mujoco
from scipy.ndimage import gaussian_filter1d
import torch

def normalize_minmax(x):
        x = np.array(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

def reward_curve(original_rs, adaptive_rs, energies):
    # Replace with your actual lists
    list1 = original_rs
    list2 = adaptive_rs
    # list3 = energies

    # Normalize each list independently to [0, 1]
    list1_norm = normalize_minmax(list1)
    list2_norm = normalize_minmax(list2)
    # list3_norm = normalize_minmax(list3)
    # list2_norm[0] = 0.613
    
    sigma = 1.0
    list2_smoothed = gaussian_filter1d(list2_norm, sigma=sigma)

    # Use index as x-axis (or replace with your own x list)
    x = range(len(list1_norm))

    plt.figure(figsize=(3.5, 1.5))

    plt.rcParams.update({
    'font.family': 'Liberation Serif',  # 主字体Times New Roman
    'font.size': 8,                    # 默认字体大小
    'axes.titlesize': 9,               # 坐标轴标题大小
    'axes.labelsize': 9,               # 坐标轴标签大小
    'xtick.labelsize': 18,              # x轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'legend.fontsize': 8,              # 图例字体大小
    'figure.titlesize': 10,            # 图标题大小
    'lines.linewidth': 1.0,            # 线宽
    'lines.markersize': 4,             # 标记大小
})

    # plt.plot(x, list1_norm, label="Original reward", linestyle='-', marker='o')
    # plt.plot(x, list2_norm, label="Adaptive reward", linestyle='--', marker='s')
    # plt.plot(x, list3_norm, label="Energy", linestyle='-.', marker='^')
    # plt.plot(x, list1_norm, label="Original reward", linestyle='--')
    # plt.plot(x, list2_norm, label="Adaptive reward", linestyle='-')
    # plt.plot(x, list3_norm, label="Energy", linestyle='-.')

    plt.plot(x, list2_smoothed, linestyle='-')

    # plt.xlabel("Step", fontsize=16, fontfamily='Times New Roman')
    # plt.ylabel("Reward value", fontsize=16, fontfamily='Times New Roman')
    # plt.xticks(fontsize=20, fontweight='normal', rotation=0)
    # plt.yticks(fontsize=20, fontweight='normal')
    # plt.xticks(np.arange(0,60,5))
    # plt.legend()
    plt.yticks([0, 1])

    plt.tight_layout()
    plt.show()
    np.save("/home/ubuntu/projects/reward_relocate_1.npy", np.array(list2_smoothed))


def reward_surface(observations, actions, reward_func):
    # obs_mean = np.mean(np.array(observations), axis=0)
    # act_mean = np.mean(np.array(actions), axis=0)
    obs_mean = observations
    act_mean = actions

    plt.rcParams.update({
    'font.family': 'Liberation Serif',  # 主字体Times New Roman
    'font.size': 8,                    # 默认字体大小
    'axes.titlesize': 9,               # 坐标轴标题大小
    'axes.labelsize': 9,               # 坐标轴标签大小
    'xtick.labelsize': 18,              # x轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'legend.fontsize': 8,              # 图例字体大小
    'figure.titlesize': 10,            # 图标题大小
    'lines.linewidth': 1.0,            # 线宽
    'lines.markersize': 4,             # 标记大小
})

    x = np.linspace(-1, 1, 200)   # 10cm 范围
    y = np.linspace(-1, 1, 200)
    Z = np.zeros((len(x), len(y)))

    for i, dx in enumerate(x):
        print(i)
        for j, dy in enumerate(y):
            s = obs_mean.copy()
            a = act_mean.copy()
            s[36] = dx  # 29
            s[37] = dy

            reward = reward_func.step(s, a)
            Z[j, i] = reward
    Z_smooth = gaussian_filter(Z, sigma=4.0)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z_smooth, cmap="magma", linewidth=0, antialiased=True)

    # ax.set_xlabel("Angular position of the IP joint of the thumb")
    # ax.set_ylabel("Angular position of the PIP joint of the forefinger")
    # ax.set_zlabel("Reward value")
    ax.tick_params(axis='x', labelsize=16, labelrotation=0)
    ax.tick_params(axis='y', labelsize=16, labelrotation=0)
    ax.tick_params(axis='z', labelsize=16, labelrotation=0)
    # ax.set_title("Reward Surface (Adroit-Relocate)")

    # cbar = fig.colorbar(surf, shrink=0.6, aspect=12)
    # cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()

def reward_surface_SFM(observations, actions, reward_func):
    obs_mean = observations
    act_mean = actions
    plt.rcParams.update({
    'font.family': 'Liberation Serif',  # 主字体Times New Roman
    'font.size': 8,                    # 默认字体大小
    'axes.titlesize': 9,               # 坐标轴标题大小
    'axes.labelsize': 9,               # 坐标轴标签大小
    'xtick.labelsize': 18,              # x轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'legend.fontsize': 8,              # 图例字体大小
    'figure.titlesize': 10,            # 图标题大小
    'lines.linewidth': 1.0,            # 线宽
    'lines.markersize': 4,             # 标记大小
})

    x = np.linspace(-1, 1, 200)   # 10cm 范围
    y = np.linspace(-1, 1, 200)
    Z = np.zeros((len(x), len(y)))

    for i, dx in enumerate(x):
        print(i)
        for j, dy in enumerate(y):
            s = obs_mean.copy()
            a = act_mean.copy()
            a[27] = dx
            a[28] = dy

            reward = reward_func.step(s, a)
            Z[j, i] = reward
    Z_smooth = gaussian_filter(Z, sigma=1)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z_smooth, cmap="viridis", linewidth=0, antialiased=True)

    # ax.set_xlabel("Angular position of the door latch")
    # ax.set_ylabel("Angular position of the door hinge")
    # ax.set_zlabel("Reward value")
    # ax.set_title("Reward Surface (Adroit-Relocate)")

    # fig.colorbar(surf, shrink=0.6, aspect=12, label="Reward")
    plt.tight_layout()
    plt.show()

def reward_surface_NEAR(obs1, obs2, reward_func):
    obs_mean = obs1
    act_mean = obs2
    plt.rcParams.update({
    'font.family': 'Liberation Serif',  # 主字体Times New Roman
    'font.size': 8,                    # 默认字体大小
    'axes.titlesize': 9,               # 坐标轴标题大小
    'axes.labelsize': 9,               # 坐标轴标签大小
    'xtick.labelsize': 18,              # x轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'ytick.labelsize': 18,              # y轴刻度标签大小
    'legend.fontsize': 8,              # 图例字体大小
    'figure.titlesize': 10,            # 图标题大小
    'lines.linewidth': 1.0,            # 线宽
    'lines.markersize': 4,             # 标记大小
})

    x = np.linspace(-1, 1, 200)   # 10cm 范围
    y = np.linspace(-1, 1, 200)
    Z = np.zeros((len(x), len(y)))

    for i, dx in enumerate(x):
        print(i)
        for j, dy in enumerate(y):
            s1 = obs1.copy()
            s2 = obs2.copy()
            s2[36] = dx
            s2[37] = dy

            reward = reward_func.step(s1, s2)
            Z[j, i] = reward
    Z_smooth = gaussian_filter(Z, sigma=1)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z_smooth, cmap="magma", linewidth=0, antialiased=True)

    # ax.set_xlabel("x positional difference from the ball to the target")
    # ax.set_ylabel("y positional difference from the ball to the target")
    # ax.set_zlabel("Reward value")
    # ax.set_title("Reward Surface (Adroit-Relocate)")

    # fig.colorbar(surf, shrink=0.6, aspect=12, label="Reward")
    plt.tight_layout()
    # plt.savefig('./reward_surface_latch_hinge_10.png', dpi=300, bbox_inches='tight')
    plt.show()

def reward_surface_original_reward(env, observations, actions, reward_func):
    # obs_mean = np.mean(np.array(observations), axis=0)
    # act_mean = np.mean(np.array(actions), axis=0)
    obs_mean = observations
    act_mean = actions

    x = np.linspace(-1, 1, 100)   # 10cm 范围
    y = np.linspace(-1, 1, 100)
    Z = np.zeros((len(x), len(y)))

    # 设置整个手为 0 rad（张开姿态）
    hand_joints = [
        "FFJ0","FFJ1","FFJ2","FFJ3",
        "MFJ0","MFJ1","MFJ2","MFJ3",
        "RFJ0","RFJ1","RFJ2","RFJ3",
        "LFJ0","LFJ1","LFJ2","LFJ3","LFJ4",
        "THJ0","THJ1","THJ2","THJ3","THJ4"
    ]

    qpos_dict = {name: 0.0 for name in hand_joints}
    set_hand_pose(env, qpos_dict)  # 先把所有关节制零

    for i, dx in enumerate(x):
        for j, dy in enumerate(y):
            s = obs_mean.copy()
            a = act_mean.copy()
            # 单独设置 joint 值
            set_hand_pose(env, {"MFJ0": dx})
            set_hand_pose(env, {"THJ0": dy})
            a = np.zeros(env.action_space.shape)
            _, reward, _, _, _ = env.step(a)  # 官方环境人工设置的reward
            # o = env.unwrapped._get_obs()
            # reward, energy = reward_func.step(o, a)
            Z[j, i] = 0

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("Angular position of the IP joint of the thumb")
    ax.set_ylabel("Angular position of the PIP joint of the forefinger")
    ax.set_zlabel("Reward")
    # ax.set_title("Reward Surface (Adroit-Relocate)")

    fig.colorbar(surf, shrink=0.6, aspect=12, label="Reward")
    plt.tight_layout()
    plt.show()
    a = 2


def set_hand_pose(env, qpos_dict):
    """
    设置 Adroit Hand 的手指关节角度。

    参数：
    - env: Gym / Gymnasium Adroit Hand 环境对象
    - qpos_dict: dict，key = joint name (e.g., "FFJ0"), value = 目标角度 (rad)

    示例：
        set_hand_pose(env, {"FFJ0": 0.1, "MFJ2": 0.5})
    """
    sim = env.unwrapped
    model = sim.model
    data = sim.data

    # 遍历 qpos_dict 中的每个关节
    for joint_name, target_angle in qpos_dict.items():
        # 获取 joint id
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        # 获取 qpos 起始索引
        addr = model.jnt_qposadr[jid]
        # 限制在合法范围内
        if joint_name == 'THJ1':
            a = 1
        low, high = model.jnt_range[jid]
        data.qpos[addr] = np.clip(target_angle, low, high)

    # 建议将速度清零，避免 reward surface 不稳定
    data.qvel[:] = 0.0

    # 更新物理状态
    mujoco.mj_forward(model, data)


def get_hand_pose(env):
    """
    获取 Adroit Hand 当前手指关节角度。

    参数：
    - env: Gym / Gymnasium Adroit Hand 环境对象

    返回：
    - dict，key = joint name (e.g., "FFJ0"), value = 当前角度 (rad)
    
    示例：
        pose = get_hand_pose(env)
        print(pose["FFJ0"])
    """
    sim = env.unwrapped
    model = sim.model
    data = sim.data

    # 手指关节名称列表（Adroit Hand 标准）
    hand_joints = [
        "FFJ0","FFJ1","FFJ2","FFJ3",
        "MFJ0","MFJ1","MFJ2","MFJ3",
        "RFJ0","RFJ1","RFJ2","RFJ3",
        "LFJ0","LFJ1","LFJ2","LFJ3","LFJ4",
        "THJ0","THJ1","THJ2","THJ3","THJ4"
    ]

    pose_dict = {}
    for joint_name in hand_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        addr = model.jnt_qposadr[jid]
        pose_dict[joint_name] = data.qpos[addr]

    return pose_dict

def env_joint_name(env):
    model = env.unwrapped.model
    data = env.unwrapped.data

    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(name)

    print(joint_names)


def reward_distribution(obs, act, rewards, device, reward_func=None, name=None):
    # 1. 准备数据
    # states: (N, d) 高维状态
    # rewards: (N,)

    if reward_func:
        # rewards = np.zeros(len(states))
        # for i in range(len(states)):
        #     r = reward_func.r(s_all).cpu().numpy()
        #     rewards[i] = r
        rewards = reward_func.step(obs, act)
    else:
        rewards = np.array(rewards)

    # states = np.concat([obs, act], axis=-1)     # 例如从 replay buffer 采样
    states = obs     # 例如从 replay buffer 采样

    # 可选：标准化（t-SNE 对尺度敏感，强烈建议）
    states = (states - states.mean(axis=0)) / (states.std(axis=0) + 1e-8)

    # 2. t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=30,        # 20~50 常用，和样本量有关
        learning_rate=200,
        max_iter=2000,
        init="pca",
        random_state=0
    )
    states_2d = tsne.fit_transform(states)   # (N, 2)

    # 3. reward 归一化（仅用于颜色）
    r_min, r_max = rewards.min(), rewards.max()
    rewards_norm = (rewards - r_min) / (r_max - r_min + 1e-8)

    # 4. 画 state–reward 分布图
    plt.figure(figsize=(6, 5))

    sc = plt.scatter(
        states_2d[:, 0],
        states_2d[:, 1],
        c=rewards_norm,
        cmap="coolwarm",      # 蓝 → 红
        s=8,
        alpha=0.8
    )

    plt.colorbar(sc)
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")
    plt.title(name)
    plt.tight_layout()
    plt.show()

def reward_distribution1(obs, act, rewards, device, reward_func=None, name=None, num_epoch=None):

    rewards = np.array(rewards)

    if num_epoch:
        ratio = len(rewards[rewards > 0]) / (rewards.size - 1000*num_epoch)

    else:
        ratio = np.mean(rewards > 0)

    return ratio




def reward_distribution_NEAR(obs1, obs2, rewards, device, reward_func=None, name=None):
    # 1. 准备数据
    # states: (N, d) 高维状态
    # rewards: (N,)

    if reward_func:
        # rewards = np.zeros(len(states))
        # for i in range(len(states)):
        #     r = reward_func.r(s_all).cpu().numpy()
        #     rewards[i] = r
        rewards = reward_func.step(obs1, obs2)
    else:
        rewards = np.array(rewards)

    # states = np.concat([obs1, obs2], axis=-1)     # 例如从 replay buffer 采样
    states = obs1     # 例如从 replay buffer 采样

    # 可选：标准化（t-SNE 对尺度敏感，强烈建议）
    states = (states - states.mean(axis=0)) / (states.std(axis=0) + 1e-8)

    # 2. t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=30,        # 20~50 常用，和样本量有关
        learning_rate=200,
        max_iter=2000,
        init="pca",
        random_state=0
    )
    states_2d = tsne.fit_transform(states)   # (N, 2)

    # 3. reward 归一化（仅用于颜色）
    r_min, r_max = rewards.min(), rewards.max()
    rewards_norm = (rewards - r_min) / (r_max - r_min + 1e-8)

    # 4. 画 state–reward 分布图
    plt.figure(figsize=(6, 5))

    sc = plt.scatter(
        states_2d[:, 0],
        states_2d[:, 1],
        c=rewards_norm,
        cmap="coolwarm",      # 蓝 → 红
        s=8,
        alpha=0.8
    )

    plt.colorbar(sc)
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")
    plt.title(name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = 0
    # 设置整个手为 0 rad（张开姿态）
    hand_joints = [
        "FFJ0","FFJ1","FFJ2","FFJ3",
        "MFJ0","MFJ1","MFJ2","MFJ3",
        "RFJ0","RFJ1","RFJ2","RFJ3",
        "LFJ0","LFJ1","LFJ2","LFJ3","LFJ4",
        "THJ0","THJ1","THJ2","THJ3","THJ4"
    ]

    qpos_dict = {name: 0.0 for name in hand_joints}
    set_hand_pose(env, qpos_dict)

    # 单独设置 FFJ0 为 0.5 rad
    set_hand_pose(env, {"FFJ0": 0.5})


    # 获取当前手指角度
    hand_pose = get_hand_pose(env)
    print("FFJ0 angle:", hand_pose["FFJ0"])
    print("Thumb angles:", [hand_pose[j] for j in ["THJ0","THJ1","THJ2","THJ3","THJ4"]])

    # 配合 set_hand_pose() 使用
    current_pose = get_hand_pose(env)
    # 例如把 FFJ0 调整 +0.2 rad
    current_pose["FFJ0"] += 0.2
    set_hand_pose(env, current_pose)