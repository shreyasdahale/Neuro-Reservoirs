import numpy as np
from sklearn.linear_model import Ridge


def sigmoid(z: np.ndarray, /) -> np.ndarray:
    """Numerically-stable logistic σ : ℝ → (0, 1)."""
    out = np.empty_like(z, dtype=np.float32)
    np.subtract(0.0, z, out)            # out = −z   (no new allocation)
    np.exp(out, out)                    # out = e^(−z)
    out += 1.0                          # 1 + e^(−z)
    np.reciprocal(out, out)             # 1 / (1 + e^(−z))
    return out


def _sample_frequencies(m: int, mode: str = "log") -> np.ndarray:
    """
    Produce m distinct rotation angles θ_i ∈ (0, π).

    * 'log'  : log-uniform in [10⁰·⁰, 10⁰·⁹] Hz then mapped to θ = 2πfΔt
    * 'lin'  : uniform in (0, π)
    * 'gold' : golden-ratio spacing (deterministic)
    """
    if mode == "log":
        # log-spread over ~1 decade, then map to (0, π)
        f = 10 ** np.linspace(0.0, 0.9, m, dtype=np.float32)
        f /= f.max()
        return np.pi * f
    if mode == "lin":
        return np.random.default_rng().uniform(0.01, np.pi - 0.01, size=m)
    if mode == "gold":
        return (np.mod(np.arange(1, m + 1) * 0.61803398875, 1.0) * (np.pi - 0.02) + 0.01)
    raise ValueError("mode must be {'log','lin','gold'}")

def augment_state_with_squares(x):
    """
    Given state vector x in R^N, return [ x, x^2, 1 ] in R^(2N+1).
    We'll use this for both training and prediction.
    """
    x_sq = x**2
    return np.concatenate([x, x_sq, [1.0]])  # shape: 2N+1


# ---------------------------------------------------------------------
# Resonator reservoir class
# ---------------------------------------------------------------------
class FSR3D:
    """
    Frequency-Selective Resonator Echo-State Network.

    • N even ⇒ m = N/2 damped 2-D rotations (planar oscillators)
    • Exact spectral control: eigenvalues r·e^{±iθ_i}
    • Sparse symmetric nearest-neighbour coupling ε·C
    • Optional static gain profile g_k  (sin→σ)  for heterogeneity
    • Optional quadratic + quadrature feature map in the read-out
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        reservoir_size: int = 400,      # N (must be even)
        frequency_mode: str = "log",    # how to draw θ_i
        r_damp: float = 0.95,           # attenuation per step
        eps_couple: float = 0.05,       # ε   cross-pair mixing strength
        input_scale: float = 0.5,
        leak_rate: float = 1.0,         # α
        ridge_alpha: float = 1e-6,
        use_gain: bool = True,
        gain_beta: float = 2.0,
        gain_sigmoid: bool = True,
        use_quadratic_feat: bool = True,
        seed: int = 42,
    ):
        if reservoir_size % 2:
            raise ValueError("reservoir_size must be even")
        self.N = reservoir_size
        self.m = reservoir_size // 2
        self.r = float(r_damp)
        self.eps = float(eps_couple)
        self.input_scale = input_scale
        self.alpha = leak_rate
        self.ridge_alpha = ridge_alpha
        self.use_gain = use_gain
        self.beta = gain_beta
        self.sig_gain = gain_sigmoid
        self.use_quad = use_quadratic_feat
        self.seed = seed
        self.freq_mode = frequency_mode

        # matrices and state
        self.W_res: np.ndarray | None = None
        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.x = np.zeros(self.N, dtype=np.float32)

        # one-off construction
        self._build_reservoir()
        self._build_gain()

    # -----------------------------------------------------------------
    # internal builders
    # -----------------------------------------------------------------
    def _build_reservoir(self):
        """
        Build   W_res = G · (R ⊕ ... ⊕ R  +  εC)   without the gain
        (gain applied separately for cheaper runtime multiplication).
        """
        m, r, eps = self.m, self.r, self.eps
        θ = _sample_frequencies(m, mode=self.freq_mode)

        # ----- block-diagonal damped rotations -----------------------
        R_blocks = np.zeros((self.N, self.N), dtype=np.float32)
        for i, theta in enumerate(θ):
            c, s = np.cos(theta) * r, np.sin(theta) * r
            i0 = 2 * i
            R_blocks[i0, i0] = c
            R_blocks[i0, i0 + 1] = -s
            R_blocks[i0 + 1, i0] = s
            R_blocks[i0 + 1, i0 + 1] = c

        # ----- sparse symmetric coupling -----------------------------
        rng = np.random.default_rng(self.seed)
        C = np.zeros_like(R_blocks)
        for i in range(m - 1):                        # nearest-neighbour chain
            b = rng.standard_normal()
            B = np.array([[b, 0.0], [0.0, b]], dtype=np.float32)
            a0, a1 = 2 * i, 2 * (i + 1)
            # upper block
            C[a0 : a0 + 2, a1 : a1 + 2] = B
            # symmetric lower block
            C[a1 : a1 + 2, a0 : a0 + 2] = B.T

        self.W_res = R_blocks + eps * C

    def _build_gain(self):
        """Static per-neuron gain g_k."""
        if not self.use_gain:
            self.g = np.ones(self.N, dtype=np.float32)
            return
        k_idx = np.arange(self.N, dtype=np.float32)
        raw = self.beta * np.sin(2.0 * np.pi * k_idx / self.N)
        self.g = sigmoid(raw) if self.sig_gain else raw.astype(np.float32)

    # -----------------------------------------------------------------
    # reservoir core
    # -----------------------------------------------------------------
    def _update(self, u_t: np.ndarray):
        pre = self.W_res @ self.x + self.W_in @ u_t
        if self.use_gain:
            pre *= self.g
            # pre = self.g*self.W_res @ self.x + self.W_in @ u_t
        
        else:
            pre = self.W_res @ self.x + self.W_in @ u_t

        new_x = np.tanh(pre)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * new_x

    def reset_state(self):
        self.x.fill(0.0)

    # -----------------------------------------------------------------
    # read-out training
    # -----------------------------------------------------------------
    def fit_readout(self, inputs: np.ndarray, targets: np.ndarray, discard: int = 100):
        """
        Teacher-forcing pass to learn W_out via ridge regression.
        * inputs  shape [T, d_in]
        * targets shape [T, d_out]
        """
        T, d_in = inputs.shape
        if T <= discard + 1:
            raise ValueError("Not enough data")

        # random input weights
        rng = np.random.default_rng(self.seed)
        self.W_in = (
            rng.uniform(-1.0, 1.0, size=(self.N, d_in)) * self.input_scale
        ).astype(np.float32)

        # collect echoed states
        self.reset_state()
        states = []
        for t in range(T):
            self._update(inputs[t])
            if t >= discard:
                states.append(self.x.copy())

        X = np.asarray(states, dtype=np.float32)            # [T−d, N]
        Y = targets[discard:]                               # align

        # feature map
        if self.use_quad:
            X_list = []
            for s in X:
                X_list.append(augment_state_with_squares(s))
            feats = np.array(X_list, dtype=np.float32)

        else:
            feats = X

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(feats, Y)
        self.W_out = reg.coef_.astype(np.float32)

    # -----------------------------------------------------------------
    # autoregressive forecast
    # -----------------------------------------------------------------
    def predict_autoregressive(self, init_u: np.ndarray, n_steps: int) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction")

        d_in = init_u.shape[0]
        preds = np.empty((n_steps, self.W_out.shape[0]), dtype=np.float32)

        #self.reset_state()
        u_t = init_u.astype(np.float32).copy()

        for t in range(n_steps):
            self._update(u_t)

            if self.use_quad:
                feat_vec = augment_state_with_squares(self.x)
            else:
                feat_vec = self.x

            y_t = (self.W_out @ feat_vec).astype(np.float32)
            preds[t] = y_t
            u_t = y_t[:d_in]

        return preds
    
    def predict_open_loop(self, inputs: np.ndarray):
        preds = []
        for true_input in inputs:
            self._update(true_input)
            if self.use_quad:
                feat_vec = augment_state_with_squares(self.x)
            else:
                feat_vec = self.x
            out = (self.W_out @ feat_vec).astype(np.float32)
            preds.append(out)
        return preds
