#!/usr/bin/env python3
"""
record_adroit.py

示例：在 Gymnasium 的 AdroitHandDoor-v1 上录制视频并保存为 mp4。
依赖：gymnasium, mujoco（若使用 MuJoCo 环境），imageio（可选）
"""
import sys, os, time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.environ["MUJOCO_GL"] = "egl"
import torch
from ruamel.yaml import YAML
import random
import pickle

from irl_methods.divs.f_div_disc import f_div_disc_loss
from irl_methods.divs.f_div import maxentirl_loss
from irl_methods.divs.ipm import ipm_loss
from irl_methods.models.reward import MLPReward
from irl_methods.models.reward_VLVRM import VLVRM, TotalRewardModule
from irl_methods.models.reward_SFM import SFMReward, sfm_loss
from irl_methods.models.reward_SFM import TotalRewardModule as TotalRewardModule_SFM
from irl_methods.models.reward_NEAR import NCSNScoreNet, get_sigmas, NEARReward, build_paired_obs, near_pretrain_loss
from irl_methods.models.reward_NEAR import TotalRewardModule as TotalRewardModule_NEAR

from irl_methods.models.discrim import SMMIRLDisc as Disc
from irl_methods.models.discrim import SMMIRLCritic as Critic
from common.sac_irl_methods_door import ReplayBuffer, SAC
# from common.sac_irl_methods_NEAR import ReplayBuffer, SAC
import envs
from utils import system, collect, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc
from utils.result_plot import reward_curve, reward_surface, reward_surface_NEAR, reward_surface_original_reward, reward_distribution, reward_distribution_NEAR, reward_distribution1
import datetime
import dateutil.tz
import json, copy
from torch.utils.tensorboard import SummaryWriter
import argparse
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import MultipleLocator
from utils.expert_dataset import ExpertDataset, LossLogger, sample_batch, get_init_state, sample_batch_withR
import minari
from datetime import datetime

def make_env(env_id, video_folder, iter, record_every=1, render_mode="rgb_array", seed=None):
    # 创建环境（指定 render_mode="rgb_array" 以便录帧）
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id, render_mode=render_mode)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    env.name_prefix = f"adroit_{now}"
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    
    class FrameInfoWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.frame_count = 0
            self.step_count = 0
            self.episode_count = 0
        def reset(self, **kwargs):
            self.step_count = 0
            self.episode_count += 1
            return super().reset(**kwargs)
        def step(self, action):
            self.step_count += 1
            return super().step(action)
        def render(self):
            frame = super().render()
            if frame is not None:
                self.frame_count += 1
                # 在帧上添加信息
                frame = self.add_frame_info(frame)
            return frame
        def add_frame_info(self, frame):
            # 复制帧以避免修改原始数据
            frame_with_text = frame.copy()
            h, w, _ = frame_with_text.shape
            # 设置字体和颜色
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)  # 白色
            thickness = 2
            # 添加文本信息
            texts = [f"Frame: {self.frame_count}",f"Step: {self.step_count}",f"Episode: {self.episode_count}"]
            # 在帧上绘制文本
            for i, text in enumerate(texts):
                y_position = 30 + i * 30
                cv2.putText(frame_with_text, text, (10, y_position), 
                           font, font_scale, color, thickness, cv2.LINE_AA)
            return frame_with_text
    
    env = FrameInfoWrapper(env)

    # episode_trigger: 控制哪些 episode 录制（此例：每 record_every 个 episode 录一次）
    episode_trigger = lambda idx: (idx % record_every == 0)

    env = RecordVideo(env,
                      video_folder=video_folder,
                      episode_trigger=episode_trigger,
                      name_prefix=f"adroit_{iter}", fps=20)
    return env


def rollout_and_record(env, agent, iter, max_episodes=5, max_steps_per_episode=1000, fps=10, seed=None):
    env = make_env(args.env, args.video_folder, iter, record_every=args.record_every, seed=args.seed)
    os.makedirs(env.video_folder, exist_ok=True) if hasattr(env, "video_folder") else None

    energies = []
    for ep in range(max_episodes):
        # ep_seed = rng.randint(0, 2**32-1)
        obs, info = env.reset(seed=seed)
        obs1 = obs  # near
        done = False
        step = 0
        ep_reward = 0.0
        original_rs, adaptive_rs, energies = [], [], []
        observations = []
        actions = []
        while not done and step < max_steps_per_episode:
            # 这里用随机策略作示例；替换为你的 policy.sample(obs) 即可
            action = agent.get_action(obs, True)
            # action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            obs2 = obs # near
            # if step == 50:
            #     reward_surface_NEAR(obs1, obs2, agent.reward_function)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            step += 1
            dist = np.abs(obs[28] - np.array(np.pi * 9. / 18.))
            obs1 = obs2 # near

            adaptive_r = agent.reward_function.step(obs, action)
            observations.append(obs)
            actions.append(action)
            original_rs.append(round(reward.item(), 4))
            adaptive_rs.append((round(adaptive_r.item(), 4)))
            # energies.append((round(energy.item(), 4)))

        print(f"Episode {ep} finished. steps={step}, total_reward={ep_reward:.3f}, dist={dist:.4f}")
        #### plot ####
        # plot reward curve
        reward_curve(original_rs, adaptive_rs, energies)
        # reward surface
        # reward_surface(observations, actions, agent.reward_function)
        # reward_surface_original_reward(env, observations, actions, agent.reward_function)

    env.close()
    print("Recording finished. Videos saved to:", os.path.abspath(env.video_folder))
    return ep_reward, dist

    # energies = energies[:55]
    # 绘制能量曲线图
    # plt.figure(figsize=(12, 6))
    # plt.plot(energies, linewidth=1, color='blue', alpha=0.7)
    # plt.xlabel('Step', fontsize=12)
    # plt.ylabel('p(s|z)', fontsize=12)
    # max_step = len(energies)
    # plt.xticks(range(0, max_step, max(1, max_step//40)))  # plt.xticks(range(0, max_step, max(1, max_step//40)))
    # plt.tick_params(axis='x', which='major', labelsize=6)
    # plt.grid(True, alpha=0.3)
    # plt.savefig('energy_curve_step200.png', dpi=300, bbox_inches='tight')


def setup_experiment_seed(seed):
    """Centralized seed setup for the entire experiment"""
    # Set basic Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Return a random number generator for generating other seeds
    return np.random.RandomState(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='f-IRL training script') # required=True,
    parser.add_argument("--env", type=str, default="AdroitHandDoor-v1", help="env id")
    parser.add_argument("--video-folder", type=str, default="./videos", help="保存视频的文件夹")
    parser.add_argument("--record-every", type=int, default=1, help="每 n 个 episode 录一次（1 表示全部录制）")
    parser.add_argument("--episodes", type=int, default=1, help="最多录制多少 episode")
    parser.add_argument("--steps", type=int, default=200, help="每个 episode 最大 step")
    parser.add_argument("--reward_func", type=str,
                        default='./logs/door-expert-v1/exp-1/maxentirl_sa/2025_12_15_00_14_46_with_OT_reglar/model/reward_model_itr190_det19169.pkl', help="每个 episode 最大 step")
    # parser.add_argument("--reward_func_noOT", type=str,
    #                     default='./logs/door-expert-v1/exp-1/maxentirl_sa/2026_03_14_10_00_42_noOT_seed42/model/reward_model_itr185_det3154.pkl', help="每个 episode 最大 step")
    # parser.add_argument("--agent_ac", type=str, 
                        # default='./logs/door-expert-v1/exp-1/maxentirl_sa/2025_12_15_00_14_46_with_OT_reglar/model/reward_model_itr190_ac.pkl', help="每个 episode 最大 step")
    parser.add_argument("--agent_ac", type=str, 
                        default='/home/ubuntu/projects/robot/SOAR-IL-test/test_weights/door/ar_110_seed45.pkl', help="每个 episode 最大 step")

    parser.add_argument("--reward_func_SFM", type=str,
                        default='./logs/door-expert-v1/exp-1/maxentirl_sa/2026_03_08_21_26_25_seed50_SFM/model/reward_model_itr193_det3102.pkl', help="每个 episode 最大 step")
    # parser.add_argument("--reward_func_NEAR", type=str,
    #                     default='near_score_model_door.pt', help="每个 episode 最大 step")
    parser.add_argument("--reward_func_NEAR", type=str,
                        default='./logs/door-expert-v1/exp-1/maxentirl_sa/NEAR/2026_03_07_19_35_01_NAER_seed50/model/near_score_model_itr277_det3102.pkl', help="每个 episode 最大 step")

    parser.add_argument('--config', type=str, default='configs/samples/agents/adroit_door.yml', help='Path to config YAML file')
    parser.add_argument('--num_q_pairs', type=int, default=1, help='Number of Q-network pairs (default: 1)')
    parser.add_argument('--seed', type=int, default=45, help='Random seed (default: from config)')
    parser.add_argument('--uncertainty_coef', type=float, default=1.0, help='Uncertainty coefficient for exploration (default: 1.0)')
    parser.add_argument('--q_std_clip', type=float, default=10.0, help='Maximum value to clip Q-value standard deviations (default: 1.0)')
    parser.add_argument('--use_hype', type=bool, default=True, help='Enable hyper parameter optimization')
    parser.add_argument('--num_epoch', type=int, default=30, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_pi', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Load config
    yaml = YAML()
    v = yaml.load(open(args.config))
    v['sac']['lr'] = args.lr
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")

    dataset = minari.load_dataset('D4RL/door/expert-v2')
    sE_all = np.concatenate([dataset[i].observations[:-1] for i in range(5000)], axis=0)  # (N_e*T, s_dim)
    aE_all = np.concatenate([dataset[i].actions for i in range(5000)], axis=0)  # (N_e*T, s_dim)
    sE_all_t = torch.FloatTensor(np.concat([sE_all, aE_all], -1))

    env_fn = lambda: gym.make(args.env)
    env = make_env(args.env, args.video_folder, 0, record_every=args.record_every, seed=args.seed)
    # env = env.unwrapped

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    state_indices = list(range(state_size))

    # reward function
    # reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    use_actions_for_reward = False
    if v['obj']=='maxentirl_sa' or v['obj']=='opt-AIL_sa' or v['obj']=='hype':
        use_actions_for_reward=True
        # reward_func = MLPReward(len(state_indices)+action_size, **v['reward'], device=device).to(device)
        reward_func = LatentEnergyReward(len(state_indices)+action_size, **v['reward'], device=device).to(device)
    state_dict = torch.load(args.reward_func, map_location='cpu')
    reward_func.load_state_dict(state_dict)

    # SFM
    reward_func_SFM = SFMReward(len(state_indices), action_size, device=device)
    state_dict_SFM = torch.load(args.reward_func_SFM, map_location='cpu')
    reward_func_SFM.load_state_dict(state_dict_SFM)
    reward_calc_SFM = TotalRewardModule_SFM(reward_func_SFM, device=device)

    # NEAR
    score_model = NCSNScoreNet(len(state_indices)*2, hidden_dim=256, num_noise_levels=10, device=device)
    state_dict_NEAR = torch.load(args.reward_func_NEAR, map_location='cpu')
    score_model.load_state_dict(state_dict_NEAR)
    sigmas = get_sigmas(1.0, 0.01, 10, device=device)
    reward_func_NEAR = NEARReward(score_model, sigmas, device=device)
    reward_calc_NEAR = TotalRewardModule_NEAR(reward_func_NEAR, device=device)

    mu_e, lv_e = reward_func.encoder(torch.tensor(sE_all_t, dtype=torch.float32).to(device))
    z_expert_cloud = mu_e.detach().cpu()

    reward_calc = TotalRewardModule(reward_func, z_expert_cloud, device=device)

    # SAC
    seed = args.seed if args.seed is not None else v['seed']
    rng = setup_experiment_seed(seed)
    env_seed = rng.randint(0, 2**32-1)
    network_seed = rng.randint(0, 2**32-1)

    num_q_pairs = args.num_q_pairs
    uncertainty_coef = args.uncertainty_coef
    q_std_clip = args.q_std_clip

    # Reset SAC agent with old policy, new environment, and new replay buffer
    print("Reinitializing sac")
    replay_buffer = ReplayBuffer(
        state_size, 
        action_size,
        device=device,
        size=v['sac']['buffer_size'])
        
    sac_agent = SAC(env_fn, replay_buffer,
        steps_per_epoch=v['env']['T'],
        update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
        max_ep_len=v['env']['max_ep_len'],
        seed=network_seed,
        start_steps=-1,
        reward_state_indices=state_indices,
        device=device,
        num_q_pairs=int(num_q_pairs),
        uncertainty_coef=uncertainty_coef,
        q_std_clip=q_std_clip, schedule=True, rl_lr_restart=False, epoch=args.num_epoch,
        use_actions_for_reward=use_actions_for_reward, lr_pi=args.lr_pi,
        opt_AIL= (v['obj'] == 'opt-AIL') or (v['obj'] == 'opt-AIL_sa'), 
        **v['sac'])
    sac_agent.logger = LossLogger()
    # sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
    # sac_agent.reward_function = reward_func.get_scalar_reward_targets
    sac_agent.reward_function = reward_calc
    
    if args.agent_ac is not None:
        # load SAC
        state_dict_ac = torch.load(args.agent_ac, map_location='cpu')
        if "model_state_dict" in state_dict_ac:
            state_dict = state_dict_ac["model_state_dict"]
        else:
            state_dict = state_dict_ac
        sac_agent.ac.load_state_dict(state_dict_ac)

    # for i in range(args.num_epoch):
    #     if args.use_hype:
    #         ex_obs, ex_acts, ex_r = sample_batch_withR(dataset, 5)
    #         # Add expert trajectories to replay buffer
    #         for ep in range(ex_obs.shape[0]):
    #             for t in range(ex_obs.shape[1]-1):
    #                 replay_buffer.store(ex_obs[ep,t], ex_acts[ep,t], 0.0, ex_obs[ep,t+1], False)
    #     print(f'Epoch_{i}')
    #     sac_info = sac_agent.learn_mujoco(print_out=True)

    #     ep_reward, similarity = rollout_and_record(env, sac_agent, i+1, max_episodes=args.episodes, max_steps_per_episode=args.steps, seed=env_seed)
    #     if ep_reward > 0 and similarity <= 0.1:
    #         folder_name = './videos/sac_agent_model'
    #         if not os.path.exists(folder_name):
    #             os.makedirs(folder_name, exist_ok=True)
    #         torch.save(sac_agent.ac.state_dict(), f"./videos/sac_agent_model/sac_agent_itr{i+1}_ac_{ep_reward}_{similarity}.pkl")
    # sac_agent.logger.save_to_file(f"logs/loss/losses_iter{i}.npz")
    
    ep_reward, similarity = rollout_and_record(env, sac_agent, 0, max_episodes=args.episodes, max_steps_per_episode=args.steps, seed=env_seed)
    # reward_func.load_state_dict(state_dict)

    # adaptive reward distribution
    # reward_distribution(replay_buffer.state[:replay_buffer.ptr],
    #                     replay_buffer.action[:replay_buffer.ptr],
    #                     replay_buffer.reward[:replay_buffer.ptr],
    #                     device, reward_func=reward_calc)
    
    # # SFM
    # reward_distribution(replay_buffer.state[:replay_buffer.ptr],
    #                     replay_buffer.action[:replay_buffer.ptr],
    #                     replay_buffer.reward[:replay_buffer.ptr],
    #                     device, reward_func=reward_calc_SFM)
    
    # # NEAR
    # reward_distribution_NEAR(replay_buffer.state[:replay_buffer.ptr],
    #                     replay_buffer.next_state[:replay_buffer.ptr],
    #                     replay_buffer.reward[:replay_buffer.ptr],
    #                     device, reward_func=reward_calc_NEAR)
    
    # # official reward distribution
    # reward_distribution(replay_buffer.state[:replay_buffer.ptr],
    #                     replay_buffer.action[:replay_buffer.ptr],
    #                     replay_buffer.reward[:replay_buffer.ptr],
    #                     device)
    
    # ratio = reward_distribution1(replay_buffer.state[:replay_buffer.ptr],
    #                     replay_buffer.action[:replay_buffer.ptr],
    #                     replay_buffer.reward[:replay_buffer.ptr],
    #                     device, num_epoch=args.num_epoch)
    # print('ratio: ', ratio)