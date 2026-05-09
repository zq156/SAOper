# conditional_vae_elbo_reward.py
# Run with: python conditional_vae_elbo_reward.py
import math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Utilities -----
def gaussian_log_prob(x, mu, logvar):
    # x, mu, logvar: (..., D)
    var = torch.exp(logvar)
    D = x.shape[-1]
    log_prob = -0.5 * ( ((x - mu)**2) / var + logvar + math.log(2*math.pi) )
    return log_prob.sum(dim=-1)

def kl_diag_normal(q_mu, q_logvar, p_mu=None, p_logvar=None):
    if p_mu is None:
        p_mu = torch.zeros_like(q_mu)
    if p_logvar is None:
        p_logvar = torch.zeros_like(q_logvar)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    term = (q_var + (q_mu - p_mu)**2) / p_var
    kl = 0.5 * ( (p_logvar - q_logvar) + term - 1.0 ).sum(dim=-1)
    return kl  # (batch,)

# Encoder: q(z|s) 潜变量z的后验
class EncoderMLP(nn.Module):
    def __init__(self, s_dim, z_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
# Decoder: p(s|z) 势能函数
class DecoderMLP(nn.Module):
    def __init__(self, z_dim, s_dim, hidden=256, fixed_logvar=-2.0, learn_var=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, s_dim)
        )
        self.learn_var = learn_var
        if learn_var:
            self.logvar = nn.Parameter(torch.tensor(fixed_logvar))
        else:
            self.register_buffer("logvar_buffer", torch.tensor(fixed_logvar))
    def forward(self, z):
        mu = self.net(z)
        if self.learn_var:
            logvar = self.logvar * torch.ones_like(mu)
        else:
            logvar = self.logvar_buffer * torch.ones_like(mu)
        return mu, logvar
    def log_prob(self, s, z):
        mu, logvar = self.forward(z)
        return gaussian_log_prob(s, mu, logvar)

# ----- VLVRM wrapper -----
class VLVRM(nn.Module):
    def __init__(self, s_dim, z_dim=32, hidden=256, prior_std=1.0, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.encoder = EncoderMLP(s_dim, z_dim, hidden)
        self.decoder = DecoderMLP(z_dim, s_dim, hidden)
        self.z_dim = z_dim
        self.prior = Normal(loc=torch.zeros(z_dim, device=device),
                            scale=torch.ones(z_dim, device=device)*prior_std)
        self.device = device
        
    def reparameterize(self, mu, logvar):
        # single-sample reparameterization for each batch element
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def q_log_prob(self, z, mu, logvar):
        return gaussian_log_prob(z, mu, logvar)  # (batch,)
    
    def prior_log_prob(self, z):
        # z: (batch, z_dim)
        return self.prior.log_prob(z).sum(-1)
    
    def elbo(self, s, print_energy=None):
        # returns ELBO estimate per element in batch (no averaging), and KL
        mu, logvar = self.encoder(s)               # (B, z), (B, z)
        z = self.reparameterize(mu, logvar)        # (B, z)
        log_psgz = self.decoder.log_prob(s, z)     # (B,)
        log_pz = self.prior_log_prob(z)            # (B,)
        log_qz = self.q_log_prob(z, mu, logvar)    # (B,)
        elbo = log_psgz + log_pz - log_qz          # (B,)
        # also compute analytic KL for monitoring: KL(q||p)
        kl = kl_diag_normal(mu, logvar)            # (B,)
        # note: elbo = log p(s|z) - KL(q||p) in expectation; here we return sample-based
        if print_energy:
            return elbo, kl, mu, logvar, z, log_psgz
        else:
            return elbo, kl, mu, logvar, z
    
    def r(self, batch):
        r, *_ = self.elbo(batch)
        r = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
        return r
    
    def r_(self, batch):
        r, *_ = self.elbo(batch)
        r = torch.clamp(r, min=-1e5, max=1e5)
        return r
    
    def r_test(self, batch):
        r, _, _, _, _, log_psgz = self.elbo(batch, print_energy=True)
        # r = torch.clamp(r, min=-1000.0, max=1000.0)
        # r = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
        return r, log_psgz
    
    def get_scalar_reward(self, obs, print_energy=None):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            if print_energy:
                r, kl, mu, logvar, z, log_psgz = self.elbo(obs, print_energy)
            else:
                r, *_ = self.elbo(obs)
            r = torch.clamp(r, min=-100000.0, max=100000.0)  
            r = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
            reward = r.cpu().detach().numpy().flatten()
            # reward_sum = reward.sum()
            # reward_std = reward.std()
        self.train()
        if print_energy:
            return reward, log_psgz
        else:
            return reward


def log_sum_exp(x, dim=None):
    """Stable log-sum-exp"""
    xmax, _ = torch.max(x, dim=dim, keepdim=True)
    return xmax + torch.log(torch.sum(torch.exp(x - xmax), dim=dim, keepdim=True))

def sinkhorn_log_domain(z_a, z_e, eps=0.05, iters=30):
    """
    Compute log-domain Sinkhorn divergence between two point clouds z_a, z_e.
    z_a: (N, d)
    z_e: (M, d)
    """
    N, d = z_a.shape
    M = z_e.shape[0]

    # cost matrix
    C = torch.cdist(z_a, z_e, p=2) ** 2  # (N, M)

    # log-Kernel = -C/eps
    K_log = -C / eps

    # dual potentials
    u = torch.zeros(N, 1, device=z_a.device)
    v = torch.zeros(M, 1, device=z_a.device)

    log_r = torch.zeros_like(u)
    log_c = torch.zeros_like(v)

    for _ in range(iters):
        u = log_r - log_sum_exp(K_log + v.T, dim=1)
        v = log_c - log_sum_exp((K_log + u).T, dim=1)

    # compute transport plan in log domain
    log_pi = K_log + u + v.T   # (N, M)

    # Sinkhorn cost
    sinkhorn_cost = torch.sum(torch.exp(log_pi) * C)

    return sinkhorn_cost

def reward_loss( agent_samples, expert_samples_obs, expert_samples_act, reward_func, device, sE_all_t,
                                     sinkhorn_eps=0.05, struct_weight=0, ot_reg_weight=1):
    """
    agent_sample_collector: function → returns (s_agent_np, a_agent_np)
    expert_dataset: list of expert (s, a) trajectories
    reward_model: LatentEnergyReward
    struct_reward: StructuredReward
    """
    sA, aA, _ = agent_samples
    sA_ = np.concatenate([sA,aA],2)
    sE = np.concatenate([expert_samples_obs, expert_samples_act], -1)

    B, T, s_dim = sA.shape

    # flatten for latent reward
    sA_flat = torch.FloatTensor(sA_.reshape(-1, sA_.shape[-1])).to(device)
    sE_flat = torch.FloatTensor(sE.reshape(-1, sA_.shape[-1])).to(device)

    r_lat_A = reward_func.r_(sA_flat).view(-1)
    r_lat_E = reward_func.r_(sE_flat).view(-1)

    with torch.no_grad():
        mu_e, lv_e = reward_func.encoder(sE_all_t)
        z_e = mu_e.detach()   # (N_e*T, z_dim)

    mu_a, lv_a = reward_func.encoder(sA_flat)
    z_a = mu_a

    # subsample latent cloud
    n = min(256, z_a.shape[0])
    m = min(256, z_e.shape[0])

    idxa = torch.randperm(z_a.shape[0])[:n]
    idxe = torch.randperm(z_e.shape[0])[:m]

    z_a_sub = z_a[idxa]
    z_e_sub = z_e[idxe]

    sink_cost = sinkhorn_log_domain(z_a_sub, z_e_sub, eps=sinkhorn_eps, iters=100)

    surrogate = r_lat_A.mean() - r_lat_E.mean()

    # MINIMIZE: -surrogate + λ * OT
    loss = surrogate + ot_reg_weight * sink_cost

    return loss

def reward_loss_noOT( agent_samples, expert_samples_obs, expert_samples_act, reward_func, device, sE_all_t,
                                     sinkhorn_eps=0.05, struct_weight=0, ot_reg_weight=1):
    """
    agent_sample_collector: function → returns (s_agent_np, a_agent_np)
    expert_dataset: list of expert (s, a) trajectories
    reward_model: LatentEnergyReward
    struct_reward: StructuredReward
    """
    sA, aA, _ = agent_samples
    sA_ = np.concatenate([sA,aA],2)
    sE = np.concatenate([expert_samples_obs, expert_samples_act], -1)

    B, T, s_dim = sA.shape

    # flatten for latent reward
    sA_flat = torch.FloatTensor(sA_.reshape(-1, sA_.shape[-1])).to(device)
    sE_flat = torch.FloatTensor(sE.reshape(-1, sA_.shape[-1])).to(device)

    r_lat_A = reward_func.r_(sA_flat).view(-1)
    r_lat_E = reward_func.r_(sE_flat).view(-1)

    with torch.no_grad():
        mu_e, lv_e = reward_func.encoder(sE_all_t)
        z_e = mu_e.detach()   # (N_e*T, z_dim)
        mu_a, lv_a = reward_func.encoder(sA_flat)
        z_a = mu_a.detach()

    # subsample latent cloud
    n = min(256, z_a.shape[0])
    m = min(256, z_e.shape[0])

    idxa = torch.randperm(z_a.shape[0])[:n]
    idxe = torch.randperm(z_e.shape[0])[:m]

    z_a_sub = z_a[idxa]
    z_e_sub = z_e[idxe]

    sink_cost = sinkhorn_log_domain(z_a_sub, z_e_sub, eps=sinkhorn_eps, iters=100)

    surrogate = r_lat_A.mean() - r_lat_E.mean()

    # MINIMIZE: -surrogate + λ * OT
    loss = surrogate + ot_reg_weight * sink_cost

    return loss

#########################################
##### 5. RL Fine-tune OT 映射
#########################################
def log_sum_exp1(x, dim):
    """stable log-sum-exp and return squeezed along dim"""
    xmax, _ = torch.max(x, dim=dim, keepdim=True)        # keepdim for stability
    lse = xmax + torch.log(torch.sum(torch.exp(x - xmax), dim=dim, keepdim=True))
    return lse.squeeze(dim)   # <-- IMPORTANT: remove the reduced dim

def compute_log_sinkhorn_pi(C, eps=0.05, max_iters=50):
    """
    C: (N, M) cost matrix (squared distances)
    returns: pi (N, M) transport plan (rows sum to 1)
    Uses log-domain Sinkhorn iterations.
    """
    device = C.device
    N, M = C.shape
    logK = - C / eps         # (N, M)
    # initialize dual variables (log-domain)
    log_u = torch.zeros(N, device=device)
    log_v = torch.zeros(M, device=device)
    # uniform marginals (log)
    log_r = -torch.log(torch.tensor(float(N), device=device))
    log_c = -torch.log(torch.tensor(float(M), device=device))

    for _ in range(max_iters):
        # update log_u, log_v using log-sum-exp
        # log_u_i = log_r_i - logsumexp_j( logK_ij + log_v_j )
        log_u = log_r - log_sum_exp1(logK + log_v.unsqueeze(0), dim=1)
        # log_v_j = log_c_j - logsumexp_i( logK_ij + log_u_i )
        log_v = log_c - log_sum_exp1((logK + log_u.unsqueeze(1)).T, dim=1)

    log_pi = logK + log_u.unsqueeze(1) + log_v.unsqueeze(0)  # (N, M)
    pi = torch.exp(log_pi)
    # normalize rows (numerical safety)
    pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-12)
    return pi

def transport_plan(z_src, z_tgt, eps=0.05, iters=50):
    """
    z_src: (n, d) torch
    z_tgt: (m, d) torch
    returns pi (n, m)
    """
    # cost matrix
    C = torch.cdist(z_src, z_tgt, p=2) ** 2
    pi = compute_log_sinkhorn_pi(C, eps=eps, max_iters=iters)
    return pi

def barycentric_map(z_src, z_tgt, eps=0.05, iters=50):
    """
    Map each z_src_i to weighted average of z_tgt via transport plan π:
      z_mapped_i = (π_i @ z_tgt) / sum(π_i)
    z_src: (n, d)
    z_tgt: (m, d)
    returns z_mapped: (n, d)
    """
    if z_src.shape[0] == 0 or z_tgt.shape[0] == 0:
        return torch.zeros_like(z_src)
    pi = transport_plan(z_src, z_tgt, eps=eps, iters=iters)  # (n,m)
    z_mapped = pi @ z_tgt  # (n, d)
    denom = pi.sum(dim=1, keepdim=True)
    z_mapped = z_mapped / (denom + 1e-12)
    return z_mapped

def compute_ot_residuals_latent(s_batch, reward_model, z_expert_cloud,
                                eps=0.05, iters=50, device='cpu'):
    """
    latent-distance residual:
      z_s <- encode(s)
      z_mapped <- barycentric_map(z_s, z_expert_cloud)
      r_resid = -|| z_s - z_mapped ||^2
    returns numpy array (n,)
    """
    reward_model.eval()
    with torch.no_grad():
        s_t = torch.as_tensor(s_batch, dtype=torch.float32, device=device)
        mu_s, logvar_s = reward_model.encoder(s_t)   # (n, z_dim)
        z_s = mu_s
        idx = torch.randperm(z_expert_cloud.size(0))[:1000]
        z_expert_sub = z_expert_cloud[idx].to(device)
        z_mapped = barycentric_map(z_s, z_expert_sub, eps=eps, iters=iters)
        diff = z_s - z_mapped
        resid = - (diff ** 2).sum(dim=-1)  # (n,)
        resid = resid.cpu().numpy()
    return resid

class StructuredRewardSimple:
    def __init__(self, g_dim=28, alpha=10.0, beta=20.0, gamma=0.1, success_thresh=0.05, device='cpu'):
        self.g_dim = g_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.success_thresh = success_thresh
        self.device = device
        self.g_t = torch.tensor(np.array(np.pi * 9. / 18.))

    def phi(self, s):
        # default: first g_dim dims encode door angle or goal
        return s[..., self.g_dim]

    def __call__(self, s_batch, a_batch=None):
        """
        s_batch: numpy array (n, s_dim) or torch tensor
        returns numpy array (n,)
        """
        if not torch.is_tensor(s_batch):
            s_t = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
        else:
            s_t = s_batch.to(self.device)

        if a_batch is not None:
            if not torch.is_tensor(a_batch):
                a_t = torch.as_tensor(a_batch, dtype=torch.float32, device=self.device)
            else:
                a_t = a_batch.to(self.device)
            energy_penalty = (a_t ** 2).sum(dim=-1) * self.gamma

        phi_s = self.phi(s_t)
        # dist = (phi_s - self.g_t)**2
        dist = torch.abs(phi_s - self.g_t)
        success_bonus = (dist < self.success_thresh).float() * self.beta
        r_struct = - self.alpha * dist + success_bonus
        return r_struct.detach().cpu().numpy()
    
def normalize(x, eps=1e-6):
    return (x - x.mean()) / (x.std() + eps)

class TotalRewardModule:
    def __init__(self, reward_model, z_expert_cloud, use_ot=True, eps=0.05, iters=50, device='cpu'):
        self.reward_model = reward_model
        self.z_expert_cloud = z_expert_cloud
        self.use_ot = use_ot
        self.eps = eps
        self.iters = iters
        self.device = device
        self.struct_reward_fn = StructuredRewardSimple()

    # def step(self, s_batch, a_batch):
    #     with torch.no_grad():
    #         s_t = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
    #         a_t = torch.as_tensor(a_batch, dtype=torch.float32, device=self.device)
    #         s_all = torch.cat([s_t, a_t], dim=-1)
    #         r_lat, energy = self.reward_model.r_test(s_all)
    #     return r_lat.detach().cpu().numpy(), energy.detach().cpu().numpy()

    def __call__(self, s_batch, a_batch):
        """
        s_batch: numpy (B, state_dim)
        a_batch: numpy (B, action_dim)
        """
        self.reward_model.eval()
        with torch.no_grad():
            s_t = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
            a_t = torch.as_tensor(a_batch, dtype=torch.float32, device=self.device)
            s_all = torch.cat([s_t, a_t], dim=-1)
            r_lat = self.reward_model.r(s_all).cpu().numpy()

        # r_struct = 0.3 * self.struct_reward_fn(s_batch, a_batch)
        # r_struct = np.maximum(r_struct, -100)
        r_struct = 0

        if self.use_ot:
            r_resid = compute_ot_residuals_latent(
                s_all, self.reward_model, self.z_expert_cloud,
                eps=self.eps, iters=self.iters, device=self.device)
        else:
            r_resid = np.zeros_like(r_lat)

        # r_lat *= 4
        r_resid_normed = 1 * normalize(r_resid)

        r_total = r_lat + r_resid_normed + r_struct

        return r_total, r_lat, r_resid, r_struct
    
    def step(self, s_batch, a_batch):
        """
        s_batch: numpy (B, state_dim)
        a_batch: numpy (B, action_dim)
        """
        self.reward_model.eval()
        with torch.no_grad():
            s_t = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
            a_t = torch.as_tensor(a_batch, dtype=torch.float32, device=self.device)
            s_all = torch.cat([s_t, a_t], dim=-1).unsqueeze(0)
            r_lat = self.reward_model.r_(s_all).cpu().numpy()

        if self.use_ot:
            r_resid = compute_ot_residuals_latent(
                s_all, self.reward_model, self.z_expert_cloud,
                eps=self.eps, iters=self.iters, device=self.device)
        else:
            r_resid = np.zeros_like(r_lat)

        # r_lat *= 4
        r_resid_normed = 1 * normalize(r_resid)

        r_total = r_lat + r_resid_normed

        return r_total