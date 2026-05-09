'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym 
from ruamel.yaml import YAML
import argparse
import random

from irl_methods.divs.f_div_disc import f_div_disc_loss
from irl_methods.divs.f_div import maxentirl_loss
from irl_methods.divs.ipm import ipm_loss
from irl_methods.models.reward import MLPReward
# from irl_methods.models.reward_LEM import LatentEnergyReward
from irl_methods.models.reward_VLVRM import VLVRM, reward_loss
from irl_methods.models.discrim import SMMIRLDisc as Disc
from irl_methods.models.discrim import SMMIRLCritic as Critic
# from common.sac import ReplayBuffer, SAC
from common.sac_irl_methods import ReplayBuffer, SAC

import envs
from utils import system, collect, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc
from utils.expert_dataset import ExpertDataset, LossLogger, sample_batch, get_init_state
import minari
import gymnasium_robotics

import datetime
import dateutil.tz
import json, copy
from torch.utils.tensorboard import SummaryWriter


def try_evaluate(itr: int, policy_type: str, sac_info, writer, global_step, seed=None):
    """Add seed parameter and pass it through to evaluation functions"""
    assert policy_type in ["Running"]
    update_time = itr * v['reward']['gradient_step']
    env_steps = itr * v['sac']['epochs'] * v['env']['T']
    agent_emp_states = samples[0].copy()
    assert agent_emp_states.shape[0] == v['irl']['training_trajs']

    # Generate evaluation seed
    eval_seed = np.random.randint(0, 2**32-1) if seed is not None else None
    
    metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                         env_steps, policy_type, seed=eval_seed)
                         
    # Pass seed to evaluation functions
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['max_ep_len'], True, seed=eval_seed)
    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    # real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
    #                                         v['irl']['eval_episodes'], v['env']['T'], False, seed=eval_seed)
    # print(f"real sto return avg: {real_return_sto:.2f}")
    # logger.record_tabular("Real Sto Return", round(real_return_sto, 2))
    
    # Log to tensorboard
    writer.add_scalar('Returns/Deterministic', real_return_det, global_step)
    # writer.add_scalar('Returns/Stochastic', real_return_sto, global_step)
    
    # Log KL metrics
    for key, value in metrics.items():
        writer.add_scalar(f'Metrics/{key}', value, global_step)

    if v['obj'] in ["emd"]:
        eval_len = int(0.1 * len(critic_loss["main"]))
        emd = -np.array(critic_loss["main"][-eval_len:]).mean()
        metrics['emd'] = emd
        logger.record_tabular(f"{policy_type} EMD", emd)
    
    # plot_disc(v['obj'], log_folder, env_steps, 
    #     sac_info, critic_loss if v['obj'] in ["emd"] else disc_loss, metrics)
    if "PointMaze" in env_name:
        visual_disc(agent_emp_states, reward_func.get_scalar_reward, disc.log_density_ratio, v['obj'],
                log_folder, env_steps, gym_env.range_lim,
                sac_info, disc_loss, metrics)

    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

    return real_return_det# , real_return_sto

def log_metrics(itr: int, sac_agent, uncertainty_coef: float, loss: float, writer: SummaryWriter, v: dict):
    """
    Log training metrics to tensorboard
    
    Args:
        itr: Current iteration number
        sac_agent: SAC agent instance
        uncertainty_coef: Uncertainty coefficient for exploration
        loss: Current reward loss value
        writer: Tensorboard SummaryWriter instance
        v: Config dictionary
    """
    # Calculate global step
    global_step = itr * v['sac']['epochs'] * v['env']['T']
    
    # Log average Q-values and their std
    q_values, q_stds = sac_agent.get_q_stats()
    writer.add_scalar('SAC/Average_Q', q_values, global_step)
    writer.add_scalar('SAC/Q_Std', q_stds, global_step)
    
    # Log reward loss
    writer.add_scalar('Training/Reward_Loss', loss.item(), global_step)
    
    return global_step

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description='f-IRL training script') # required=True,
    parser.add_argument('--config', type=str, default='configs/samples/agents/adroit_door.yml',
                      help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: from config)')
    parser.add_argument('--uncertainty_coef', type=float, default=1.0,
                      help='Uncertainty coefficient for exploration (default: 1.0)')
    parser.add_argument('--q_std_clip', type=float, default=10.0,
                      help='Maximum value to clip Q-value standard deviations (default: 1.0)')
    parser.add_argument('--use_hype', type=bool, default=True, help='Enable hyper parameter optimization')

    args = parser.parse_args()

    # Load config
    yaml = YAML()
    v = yaml.load(open(args.config))

    # assumptions
    assert v['obj'] in ['maxentirl','maxentirl_sa', 'hype']
    assert v['IS'] == False
    
    # Use parsed arguments
    seed = args.seed if args.seed is not None else v['seed']
    uncertainty_coef = args.uncertainty_coef
    q_std_clip = args.q_std_clip

    print("seed:", seed)
    print("uncertainty_coef:", uncertainty_coef)
    print("q_std_clip:", q_std_clip)
    print("obj:", v['obj'])

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    
    # Setup main experiment seed
    seed = args.seed if args.seed is not None else v['seed']
    rng = setup_experiment_seed(seed)
    
    # Generate separate seeds for different components
    env_seed = rng.randint(0, 2**32-1)
    buffer_seed = rng.randint(0, 2**32-1)
    network_seed = rng.randint(0, 2**32-1)
    
    system.reproduce(seed)
    pid=os.getpid()


    # TODO: change back 
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}" # task/obj/date structure
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S') + f'_seed{seed}' + f'_qstd{q_std_clip}'
    logger.configure(dir=log_folder)            
    writer = SummaryWriter(log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp firl/irl_samples.py {log_folder}')
    os.system(f'cp {args.config} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    dataset = minari.load_dataset('D4RL/door/expert-v2')
    sE_all = np.concatenate([dataset[i].observations[:-1] for i in range(len(dataset))], axis=0)  # (N_e*T, s_dim)
    aE_all = np.concatenate([dataset[i].actions for i in range(len(dataset))], axis=0)  # (N_e*T, s_dim)
    sE_all_t = torch.FloatTensor(np.concat([sE_all, aE_all], -1)).to(device)

    door_dataset = ExpertDataset(dataset)
    dataloader = DataLoader(
        door_dataset,
        batch_size=v['irl']['training_trajs'],
        shuffle=True,
        drop_last=True
    )

    # environment
    env_name = 'AdroitHandDoor-v1'
    gym.register_envs(gymnasium_robotics)
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    gym_env.reset(seed=env_seed)  # Seed the main environment
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    expert_trajs = dataset[0].observations
    expert_samples = expert_trajs
    print(expert_trajs.shape, expert_samples.shape) # ignored starting state

    # load expert actions
    expert_a = dataset[0].actions
    expert_a_samples = expert_a.copy().reshape(-1, action_size)
    expert_samples_sa=np.concatenate([expert_samples[:-1],expert_a_samples],1)
    print(expert_trajs.shape, expert_samples_sa.shape) # ignored starting state

    # Initilialize reward as a neural network
    use_actions_for_reward = False
    if v['obj']=='maxentirl_sa' or v['obj']=='opt-AIL_sa' or v['obj']=='hype':
        use_actions_for_reward=True
        reward_func = VLVRM(len(state_indices)+action_size, **v['reward'], device=device).to(device)

    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=v['reward']['lr'], 
        weight_decay=v['reward']['weight_decay'], betas=(v['reward']['momentum'], 0.999))

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    # for itr in range(v['irl']['n_itrs']):
    itr = 0
    for i in range(8):
        for batch in dataloader:
            expert_samples_obs = batch['obs'][:,:-1]
            expert_samples_acts = batch['acts']

            if v['sac']['reinitialize'] or itr == 0:
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
                    max_ep_len=v['env']['max_ep_len'], # ['max_ep_len']
                    seed=network_seed,
                    start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                    reward_state_indices=state_indices,
                    device=device,
                    num_q_pairs=1,
                    uncertainty_coef=uncertainty_coef,
                    q_std_clip=q_std_clip,
                    use_actions_for_reward=use_actions_for_reward, schedule=True, rl_lr_restart=True, epoch=v['irl']['n_itrs'],
                    opt_AIL= (v['obj'] == 'opt-AIL') or (v['obj'] == 'opt-AIL_sa'), 
                    **v['sac']
                )
                sac_agent.logger = LossLogger()
            
            if args.use_hype:
                ex_obs, ex_acts = sample_batch(dataset, 5)
                # Add expert trajectories to replay buffer
                for ep in range(ex_obs.shape[0]):
                    for t in range(ex_obs.shape[1]-1):
                        replay_buffer.store(ex_obs[ep,t], ex_acts[ep,t], 0.0, ex_obs[ep,t+1], False)
            
            sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
            sac_info = sac_agent.learn_mujoco(print_out=True)

            start = time.time()
            samples = collect.collect_trajectories_policy_single(gym_env, sac_agent, 
                            n = v['irl']['training_trajs'], state_indices=state_indices)
            # Fit a density model using the samples
            agent_emp_states = samples[0].copy()
            agent_emp_states = agent_emp_states.reshape(-1,agent_emp_states.shape[2]) # n*T states
            print(f'collect trajs {time.time() - start:.0f}s', flush=True)

            start = time.time()

            # optimization w.r.t. reward
            reward_losses = []
            for _ in range(v['reward']['gradient_step']):
                if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy() # resampling the expert trajectories
                elif v['irl']['resample_episodes'] > 0:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy()
                else:
                    expert_trajs_train = None # not use expert trajs

                if v['obj'] == 'maxentirl_sa':
                    expert_samples_sa=np.concatenate([expert_samples_obs,expert_samples_acts],-1)
                    loss = reward_loss(samples, expert_samples_obs, expert_samples_acts, reward_func, device, sE_all_t)
                
                reward_losses.append(loss.item())
                print(f"{v['obj']} loss: {loss}")
                reward_optimizer.zero_grad()
                loss.backward()
                reward_optimizer.step()

            # Log metrics and get global step
            global_step = log_metrics(itr, sac_agent, uncertainty_coef, loss, writer, v)
            
            # evaluating the learned reward
            real_return_det = try_evaluate(itr, "Running", sac_info, writer, global_step, seed=seed)
            if real_return_det > 1000:
                max_real_return_det = real_return_det

                torch.save(sac_agent.ac.state_dict(), os.path.join(logger.get_dir(), 
                        f"model/reward_model_itr{itr}_ac.pkl"))

                torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), 
                        f"model/reward_model_itr{itr}_det{max_real_return_det:.0f}.pkl"))

            logger.record_tabular("Itration", itr)
            logger.record_tabular("Reward Loss", loss.item())
            if v['sac']['automatic_alpha_tuning']:
                logger.record_tabular("alpha", sac_agent.alpha.item())

            logger.dump_tabular()
            itr += 1

    writer.close()