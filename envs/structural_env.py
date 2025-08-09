import numpy as np
import gymnasium as gym
from gymnasium import spaces

class StructuralEnv(gym.Env):
    """
    Multi-agent SDOF stack (one DOF per 'agent') in relative coordinates.
    Dynamics per agent i:
        m_i * xdd_i + c_i * xd_i + k_i * x_i = u_i  - m_i * a_g(t)
    where x is inter-story relative displacement; a_g is base (ground) accel.
    Integration: Newmark-β (average acceleration: β=1/4, γ=1/2).
    Actions are tendon forces u (gated by the ASCE soft gate).
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 n_agents: int = 3,
                 obs_dim: int = 6,
                 act_dim: int = 1,
                 dt: float = 0.02,
                 ep_len: int = 500,
                 seed: int | None = None,
                 # Newmark parameters
                 beta: float = 1/4,
                 gamma: float = 1/2,
                 # Reward weights
                 w_drift: float = 5.0,
                 w_accel: float = 1.0,
                 w_effort: float = 1e-4,
                 ag_std: float = 1.5,

                 ):
        super().__init__()
        self.n_agents, self.obs_dim, self.act_dim = n_agents, obs_dim, act_dim
        self.dt, self.ep_len, self.t = dt, ep_len, 0
        self.beta, self.gamma = float(beta), float(gamma)
        self.w_drift, self.w_accel, self.w_effort = w_drift, w_accel, w_effort
        self.ag_std = ag_std

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_agents, act_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_agents, obs_dim), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self._build_baseline_params()

        self.state = None  # (n_agents, 3): [x, v, a]
        self._last_u = np.zeros((n_agents, act_dim), dtype=np.float32)
        self._ag = 0.0  # last ground accel sample

    # ---------------- parameters & gate ----------------
    def _build_baseline_params(self):
        n = self.n_agents
        # Roughly increasing stiffness up the height
        self.m = np.ones(n, dtype=np.float32) * 1.0
        self.k = np.linspace(10.0, 30.0, n).astype(np.float32)
        # ~5% critical damping per DOF
        self.c = 0.05 * (2.0 * np.sqrt(self.k * self.m))

        # Code thresholds (examples—replace with your spreadsheet values)
        self.drift_lim = np.ones(n, dtype=np.float32) * 0.02  # story drift ratio (here treat x as already a ratio or scale x accordingly)
        self.accel_lim = np.ones(n, dtype=np.float32) * 0.5   # m/s^2

        # Soft ASCE gate (meta‑tunable later)
        self.alpha, self.beta_gate, self.kappa, self.hyst = 1.0, 1.0, 10.0, 0.05
        self._gate_state = np.zeros(n, dtype=np.float32)

    def _soft_gate(self, x, v, a):
        rd = np.abs(x) / (self.drift_lim + 1e-9)
        ra = np.abs(a) / (self.accel_lim + 1e-9)
        z = self.alpha * (rd - 1.0) + self.beta_gate * (ra - 1.0)
        theta_on, theta_off = 0.0 + self.hyst, 0.0 - self.hyst
        target = (z > theta_on).astype(np.float32)
        target = np.where(z < theta_off, 0.0, target)
        self._gate_state = 0.9 * self._gate_state + 0.1 * target
        gate = 1.0 / (1.0 + np.exp(-self.kappa * (z + 0.1 * self._gate_state)))
        return gate  # (n,)

    # --------------- Newmark‑β step (average acceleration) ---------------
    def _newmark_step(self, x, v, a, u, ag):
        """
        One Δt step for SDOF per agent using Newmark average-acceleration.
        Equation: m xdd + c xd + k x = u - m ag
        """
        dt = self.dt
        beta, gamma = self.beta, self.gamma
        m, c, k = self.m, self.c, self.k

        # Predictors
        x_pred = x + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a

        # Effective stiffness and force per agent (scalar SDOF each)
        k_eff = k + (m / (beta * dt * dt)) + (c * (gamma / (beta * dt)))
        # external effective force: control minus base inertia
        p_ext = (u.squeeze(-1) - m * ag)

        # Effective load uses x_pred coupling terms
        p_eff = p_ext + m * (x_pred / (beta * dt * dt)) + c * (gamma * x_pred / (beta * dt))

        # Solve per-agent (elementwise divide since SDOFs are uncoupled)
        x_next = p_eff / (k_eff + 1e-9)
        a_next = (x_next - x_pred) / (beta * dt * dt)
        v_next = v_pred + dt * gamma * a_next

        return x_next.astype(np.float32), v_next.astype(np.float32), a_next.astype(np.float32)

    # --------------- base excitation ---------------
    def _sample_ag(self):
        """
        Simple band-limited white noise as ground acceleration (m/s^2).
        Replace later with real records or a loader.
        """
        # AR(1) filter to smooth the noise
        rho = 0.98
        self._ag = rho * self._ag + (1 - rho) * self.ag_std * self.rng.standard_normal()
        return np.float32(self._ag)

    # ---------------- gym API ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        n = self.n_agents
        x0 = 0.002 * self.rng.standard_normal(n).astype(np.float32)
        v0 = np.zeros(n, dtype=np.float32)
        a0 = np.zeros(n, dtype=np.float32)
        self.state = np.stack([x0, v0, a0], axis=-1)
        self._gate_state[:] = 0.0
        self._last_u[:] = 0.0
        self._ag = 0.0
        return self._build_obs(), {}

    def _build_obs(self):
        x, v, a = self.state[:, 0], self.state[:, 1], self.state[:, 2]
        zeros = np.zeros_like(x)
        # [x, v, a, x_ref, v_ref, a_ref] (refs zero for now)
        return np.stack([x, v, a, zeros, zeros, zeros], axis=-1).astype(np.float32)

    def step(self, action):
        self.t += 1
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_agents, self.act_dim):
            raise ValueError(f"action shape {action.shape} != {(self.n_agents, self.act_dim)}")

        # Gate tendon forces by ASCE thresholds
        x, v, a = self.state[:, 0], self.state[:, 1], self.state[:, 2]
        gate = np.ones((self.n_agents, 1), dtype=np.float32)  # TEMP: always ON
        u = gate * action
        self._last_u = u.copy()

        # Base excitation
        ag = self._sample_ag()  # scalar; broadcast to all DOFs

        # Newmark‑β update
        x, v, a = self._newmark_step(x, v, a, u, ag)
        self.state = np.stack([x, v, a], axis=-1)

        # -------- reward (ASCE‑aware) --------
        rd = (np.abs(x) / (self.drift_lim + 1e-9))         # drift ratio normalized
        ra = (np.abs(a) / (self.accel_lim + 1e-9))         # accel normalized
        effort = (u.squeeze(-1))**2

        r = - (self.w_drift * rd**2 + self.w_accel * ra**2 + self.w_effort * effort).astype(np.float32)

        terminated = np.zeros(self.n_agents, dtype=bool)
        truncated  = np.array([self.t >= self.ep_len] * self.n_agents, dtype=bool)
        info = {"gate": gate.squeeze(-1), "u": u.squeeze(-1), "ag": ag}
        return self._build_obs(), r, bool(terminated.any()), bool(truncated.any()), info

    def render(self): pass
    def close(self): pass
