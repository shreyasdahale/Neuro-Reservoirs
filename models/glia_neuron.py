import numpy as np
from sklearn.linear_model import Ridge


class GliaNeuronTripartiteReservoirESN:
    """
    Glia–Neuron Tripartite-Synapse Reservoir (GNT-SR)

    ------------------------------------------------------------------
    State variables
        x  ∈ ℝᴺ   : neuronal membrane potentials (fast)
        c  ∈ ℝᴹ   : astrocytic Ca²⁺ concentrations   (intermediate)
        g  ∈ ℝᴹ   : gliotransmitter release fractions (slow)

    Update order per time-step t → t+1
        1)  c     ←   (1-μ)·c   + μ·η·σ(W_ng x − θ)   + μ·D·L_g c
        2)  g     ←   γ·c / (1 + c)
        3)  x     ←  (1-α)·x + α·tanh(W_nn x + W_in u + W_gn g)

    Echo-state property is guaranteed by fixing ρ(W_nn) < 1 and because
    astrocytic modulation acts only as an added bias.
    """

    # ------------------------------------------------------------------
    #                          constructor                              
    # ------------------------------------------------------------------
    def __init__(
        self,
        reservoir_size: int = 800,      # N  neurons
        n_astrocytes: int | None = None,  # M  astrocytes; default N//10
        input_dim: int = 3,
        rho_star: float = 0.8,          # desired spectral radius of W_nn
        alpha: float = 0.5,             # neuronal leak        (α)
        tau_c: float = 200.0,           # Ca²⁺ relaxation time (steps)
        eta: float = 1.0,               # Ca²⁺ activation gain (η)
        D_diff: float = 0.005,          # astrocytic diffusion (D)
        theta: float = 0.0,             # Ca²⁺ threshold       (θ)
        gamma_glia: float = 0.4,        # max gliotransmitter  (γ)
        input_scale: float = 0.5,
        ridge_alpha: float = 1e-6,
        use_quadratic_readout: bool = True,
        seed: int = 42,
    ):
        # -------------- dimensions ------------------------------------
        self.N = reservoir_size
        self.M = n_astrocytes if n_astrocytes is not None else max(1, reservoir_size // 10)
        self.d_in = input_dim

        # -------------- hyper-parameters ------------------------------
        self.rho_star = rho_star
        self.alpha = alpha
        self.mu = 1.0 / tau_c           # μ = Δt / τ_c  (Δt = 1)
        self.eta = eta
        self.D = D_diff
        self.theta = theta
        self.gamma = gamma_glia
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        self.use_quad = use_quadratic_readout
        self.seed = seed

        rng = np.random.default_rng(self.seed)

        # --------------------------------------------------------------
        # static weight matrices
        # --------------------------------------------------------------
        # 1) Neuron-neuron recurrent matrix   W_nn
        W_raw = rng.standard_normal((self.N, self.N)).astype(np.float32)
        eig_max = np.max(np.abs(np.linalg.eigvals(W_raw)))
        self.W_nn = (self.rho_star / eig_max) * W_raw

        # 2) Input weights  W_in
        self.W_in = (
            rng.uniform(-1.0, 1.0, size=(self.N, self.d_in)) * self.input_scale
        ).astype(np.float32)

        # 3) Neuron → astrocyte  W_ng   (rows sparse, non-neg)
        self.W_ng = rng.uniform(0.0, 1.0, size=(self.M, self.N)).astype(np.float32)
        # 4) Astrocyte → neuron  W_gn   (columns sparse, non-neg)
        self.W_gn = rng.uniform(0.0, 1.0, size=(self.N, self.M)).astype(np.float32)

        # 5) Laplacian L_g  for astrocyte diffusion (1-D chain for simplicity)
        L = np.zeros((self.M, self.M), dtype=np.float32)
        for i in range(self.M):
            if i > 0:
                L[i, i - 1] = -1.0
            if i < self.M - 1:
                L[i, i + 1] = -1.0
            L[i, i] = (2.0 if 0 < i < self.M - 1 else 1.0)
        self.L_g = L

        # --------------------------------------------------------------
        # dynamic state vectors
        # --------------------------------------------------------------
        self.x = np.zeros(self.N, dtype=np.float32)
        self.c = np.zeros(self.M, dtype=np.float32)
        self.g = np.zeros(self.M, dtype=np.float32)

        self.W_out: np.ndarray | None = None

    # ------------------------------------------------------------------
    #                            helpers                                
    # ------------------------------------------------------------------
    def reset_state(self):
        self.x.fill(0.0)
        self.c.fill(0.0)
        self.g.fill(0.0)

    def _step(self, u_t: np.ndarray):
        """One full tripartite update."""
        # ---- astrocytic Ca²⁺ -----------------------------------------
        pre_c = self.W_ng @ self.x - self.theta
        sigma_term = 1.0 / (1.0 + np.exp(-pre_c))        # logistic σ
        diff_term = self.D * (self.L_g @ self.c)
        self.c = (1.0 - self.mu) * self.c + self.mu * (self.eta * sigma_term + diff_term)

        # ---- gliotransmitter release --------------------------------
        self.g = self.gamma * self.c / (1.0 + self.c)    # element-wise

        # ---- neuronal update ----------------------------------------
        bias = self.W_gn @ self.g                        # N-vector
        pre = self.W_nn @ self.x + self.W_in @ u_t + bias
        x_new = np.tanh(pre).astype(np.float32)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * x_new

    # ------------------------------------------------------------------
    #                       read-out training                           
    # ------------------------------------------------------------------
    def fit_readout(self, inputs: np.ndarray, targets: np.ndarray, discard: int = 100):
        """
        Teacher forcing to train W_out.

        inputs  : [T, d_in]
        targets : [T, d_out]
        """
        T, d_in = inputs.shape
        if d_in != self.d_in:
            raise ValueError("input_dim mismatch")
        if T <= discard + 1:
            raise ValueError("sequence too short")

        self.reset_state()
        feats_list = []
        for t in range(T):
            self._step(inputs[t])
            if t >= discard:
                if self.use_quad:
                    feats = np.concatenate(
                        [
                            self.x,
                            self.x * self.x,
                            self.c,
                            self.g,
                            [1.0],
                        ]
                    )
                else:
                    feats = np.concatenate([self.x, self.c, self.g, [1.0]])
                feats_list.append(feats)

        X_feat = np.asarray(feats_list, dtype=np.float32)
        Y = targets[discard:]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_feat, Y)
        self.W_out = reg.coef_.astype(np.float32)

    # ------------------------------------------------------------------
    #                     autoregressive rollout                        
    # ------------------------------------------------------------------
    def predict_autoregressive(self, init_input: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Free-run the reservoir for n_steps.

        init_input : shape (d_in,)
        returns    : shape (n_steps, d_out)
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() first")

        d_out = self.W_out.shape[0]
        preds = np.empty((n_steps, d_out), dtype=np.float32)

        #self.reset_state()
        u_t = init_input.astype(np.float32).copy()

        for t in range(n_steps):
            self._step(u_t)

            if self.use_quad:
                feat_vec = np.concatenate([self.x, self.x * self.x, self.c, self.g, [1.0]])
            else:
                feat_vec = np.concatenate([self.x, self.c, self.g, [1.0]])

            y_t = (self.W_out @ feat_vec).astype(np.float32)
            preds[t] = y_t
            u_t = y_t[: self.d_in]      # feedback first d_in components

        return preds
