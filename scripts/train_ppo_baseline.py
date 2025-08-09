# scripts/train_ppo_baseline.py
# PPO baseline for StructuralEnv — robust action/reward shapes, TensorBoard logs, checkpoints.
# Run long: 8192 steps/epoch × 100 epochs. CUDA if available.

import os, sys, time
import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# Ensure project root on sys.path so `envs` works even if run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.structural_env import StructuralEnv  # noqa: E402

# --- Device & helpers ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_t(x):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def as_scalar(x) -> float:
    """Coerce reward-like objects (tensor/array/python) to a float scalar.
    If vector reward is given, use mean (baseline only)."""
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
    else:
        a = np.asarray(x)
    if a.size == 1:
        return float(a.reshape(()))
    return float(a.astype(np.float32).mean())


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_gae(rew, val, term, trunc, gamma=0.995, lam=0.95):
    T = len(rew)
    adv = torch.zeros(T, device=device)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(term[t] or trunc[t])
        nextv = val[t + 1] if t < T - 1 else 0.0
        delta = rew[t] + gamma * nonterminal * nextv - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + val
    return adv, ret


def _reset(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out if isinstance(out, tuple) and len(out) == 2 else (out, {})


def main():
    # ---- env ----
    env = StructuralEnv()
    assert isinstance(env.observation_space, gym.spaces.Box), "obs must be gym.spaces.Box"
    assert isinstance(env.action_space, gym.spaces.Box), "act must be gym.spaces.Box"

    obs_dim = int(np.prod(env.observation_space.shape))

    # Action shape & bounds — robust to (n,), (n,1), (1,n)
    sample_act = env.action_space.sample()
    target_shape = np.asarray(sample_act).shape
    act_dim = int(np.prod(target_shape))
    low = np.broadcast_to(env.action_space.low, target_shape).astype(np.float32)
    high = np.broadcast_to(env.action_space.high, target_shape).astype(np.float32)
    flat_low, flat_high = to_t(low).flatten(), to_t(high).flatten()

    # ---- nets ----
    policy = MLP(obs_dim, act_dim).to(device)
    value = MLP(obs_dim, 1).to(device)
    log_std = nn.Parameter(torch.zeros(act_dim, device=device))

    opt_pi = torch.optim.Adam(list(policy.parameters()) + [log_std], lr=3e-4)
    opt_v = torch.optim.Adam(value.parameters(), lr=3e-4)

    # ---- hyperparams (overnight) ----
    steps_per_epoch = 8192
    epochs = 100
    clip_eps = 0.2
    train_epochs = 4
    minibatches = 8
    max_ep_len = 2000
    gamma = 0.995
    lam = 0.95
    entropy_coef = 0.01

    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/tb")

    global_step = 0
    for epoch in range(epochs):
        obs_buf, act_buf, logp_buf, rews, vals = [], [], [], [], []
        term_buf, trunc_buf = [], []
        ep_rets = []

        obs, info = _reset(env, seed=int(time.time()) % 2**16)
        ep_ret, ep_len = 0.0, 0

        # ---- rollout ----
        for _ in range(steps_per_epoch):
            ot = to_t(obs).reshape(1, -1)
            with torch.no_grad():
                mu = policy(ot)                            # [1, act_dim]
                std = log_std.exp().expand_as(mu)
                dist = Normal(mu, std)
                a = dist.rsample()                         # [1, act_dim]
                logp = dist.log_prob(a).sum(-1)            # [1]
                v = value(ot).squeeze(-1)                  # [1]

            a_flat = torch.clamp(a.squeeze(0), flat_low, flat_high).cpu().numpy()
            a_np = a_flat.reshape(target_shape)

            step = env.step(a_np)
            if len(step) == 5:
                obs2, r, term, trunc, info = step
            else:
                obs2, r, term, info = step; trunc = False

            r_scalar = as_scalar(r)

            # store
            obs_buf.append(ot.squeeze(0))
            act_buf.append(a.squeeze(0))
            logp_buf.append(logp.squeeze(0))
            rews.append(torch.tensor(r_scalar, dtype=torch.float32, device=device))
            vals.append(v.squeeze(0))
            term_buf.append(term); trunc_buf.append(trunc)

            ep_ret += r_scalar
            ep_len += 1
            global_step += 1

            if term or trunc or ep_len >= max_ep_len:
                ep_rets.append(ep_ret)
                obs, info = _reset(env)
                ep_ret, ep_len = 0.0, 0
            else:
                obs = obs2

        # ---- batch tensors ----
        N = len(rews)
        obs_t  = torch.stack(obs_buf)
        act_t  = torch.stack(act_buf)
        logp_t = torch.stack(logp_buf)
        rew_t  = torch.stack(rews)
        val_t  = torch.stack(vals)
        term_t = torch.tensor(term_buf,  dtype=torch.bool, device=device)
        trunc_t= torch.tensor(trunc_buf, dtype=torch.bool, device=device)

        with torch.no_grad():
            adv_t, ret_t = compute_gae(rew_t, val_t, term_t, trunc_t, gamma=gamma, lam=lam)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ---- PPO updates ----
        idx = torch.randperm(N, device=device)
        splits = torch.chunk(idx, minibatches)
        for _ in range(train_epochs):
            for sp in splits:
                b_obs = obs_t[sp]
                b_act = act_t[sp]
                b_old = logp_t[sp].detach()
                b_adv = adv_t[sp]
                b_ret = ret_t[sp]

                mu = policy(b_obs)
                std = log_std.exp().expand_as(mu)
                dist = Normal(mu, std)
                new = dist.log_prob(b_act).sum(-1)

                ratio = torch.exp(new - b_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                pi_loss = -torch.min(surr1, surr2).mean()

                v_pred = value(b_obs).squeeze(-1)
                v_loss = 0.5 * (v_pred - b_ret).pow(2).mean()

                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + 0.5 * v_loss - entropy_coef * ent

                opt_pi.zero_grad(set_to_none=True)
                opt_v.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), 0.5)
                opt_pi.step(); opt_v.step()

        mean_ret = float(np.mean(ep_rets)) if ep_rets else float(rew_t.mean().cpu())
        print(f"iter {epoch:03d} | mean_ep_return {mean_ret: .4f} | steps {global_step}")
        writer.add_scalar("train/mean_ep_return", mean_ret, epoch)

        # checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "policy": policy.state_dict(),
                "value": value.state_dict(),
                "log_std": log_std.detach().cpu(),
                "epoch": epoch + 1,
            }, f"runs/checkpoint_e{epoch+1:03d}.pt")

    writer.close()
    print("TRAINING_BASELINE_DONE")


if __name__ == "__main__":
    main()
