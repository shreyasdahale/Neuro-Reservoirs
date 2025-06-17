import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def scale_spectral_radius(W, target_radius=0.95):
    """
    Scales a matrix W so that its largest eigenvalue magnitude = target_radius.
    """
    eigvals = np.linalg.eigvals(W)
    radius = np.max(np.abs(eigvals))
    if radius == 0:
        return W
    return (W / radius) * target_radius

def augment_state_with_squares(x):
    """
    Given state vector x in R^N, return [ x, x^2, 1 ] in R^(2N+1).
    We'll use this for both training and prediction.
    """
    x_sq = x**2
    return np.concatenate([x, x_sq, [1.0]])  # shape: 2N+1

class SAR3D:
    """
    Self-Attention Reservoir (SAR) for 3D->3D single-step tasks.
    We maintain:
      - A base reservoir_size dimension
      - random queries Qi and keys Kj for each node i,j (embedding_dim)
      - data-driven adjacency a_ij(t) = softmax_j( Q_i^T * K_j * x_j(t)? ).

    We'll do:
      s_{ij} = Q_i^T K_j * x_j(t)
      a_{ij}(t) = softmax over j of s_{ij}
      Then x_i(t+1) = (1-alpha)*x_i(t) + alpha*tanh( sum_j a_{ij}(t)*x_j(t) + W_in_i * u(t) ).

    We keep Q,K random so we only train readout in ESN style.
    """
    def __init__(self,
                 reservoir_size=300,
                 embedding_dim=16,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):
        self.reservoir_size = reservoir_size
        self.embedding_dim = embedding_dim
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        # We'll keep a base adjacency for fallback or partial usage
        # (But we can set it to small scale or keep it for optional usage. We'll decide later.)
        np.random.seed(self.seed)
        W_base = np.random.randn(reservoir_size, reservoir_size)*0.1
        W_base = scale_spectral_radius(W_base, self.spectral_radius*0.4)  # 0.0 => no static adjacency for now
        self.W_base = W_base  # possibly we won't use it 

        # Input weights
        np.random.seed(self.seed+1)
        self.W_in = (np.random.rand(reservoir_size,3) - 0.5)*2.0*self.input_scale

        # Query/Key embeddings for each node i
        np.random.seed(self.seed+2)
        self.Q = np.random.randn(reservoir_size, embedding_dim)*0.1  # Qi in R^embedding_dim
        np.random.seed(self.seed+3)
        self.K = np.random.randn(reservoir_size, embedding_dim)*0.1  # Kj in R^embedding_dim

        self.x = np.zeros(reservoir_size)
        self.W_out = None

    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)

    def _compute_attention(self, x_vec):
        """
        x_vec shape => (reservoir_size,)
        We'll produce a dynamic adjacency a_{ij} using:
          s_{ij} = (Q_i dot K_j) * x_j
        Then row-wise softmax over j for each i.
        We'll get a matrix a(t) shape => (N, N).
        """
        N = self.reservoir_size
        # compute s_{ij}
        # Q_i dot K_j => shape => (N, N)
        # then multiply by x_j(t)
        # We'll do outer product Q*K => shape (N,N), then multiply columns by x_j
        # we can do it in vectorized form:
        # Q => shape (N, embedding_dim), K => shape (N, embedding_dim)
        # Q*K^T => shape (N,N) => s_base
        s_base = self.Q @ self.K.T   # shape (N, N)
        # then s_{ij} = s_base_{ij} * x_j
        # We'll broadcast x_j across columns. s_ij = s_base_{ij} * x_j(t)
        # So for row i, we multiply each column j by x_j.
        # s(t) => shape (N, N)
        s_mat = s_base * x_vec[None,:]

        # Now row-wise softmax
        # for each i, a_i(t) = softmax( s_mat[i,:] ), shape => (N,)
        a = np.zeros((N,N))
        for i in range(N):
            row = s_mat[i,:]
            row_max = np.max(row)
            exp_row = np.exp(row - row_max)
            sum_exp = np.sum(exp_row)
            if sum_exp < 1e-12:
                a[i,:] = 0.0
            else:
                a[i,:] = exp_row / sum_exp

        return a  # shape => (N,N)

    def _update(self, u):
        """
        x(t+1) = (1-alpha)*x(t) + alpha*tanh( sum_j a_{ij}(t)* x_j(t) + W_in_i * u(t) ).
        We'll ignore W_base or set it to zero if we want purely attention-based adjacency.
        """
        a_mat = self._compute_attention(self.x)  # shape => (N,N)

        # sum_j a_ij(t)* x_j(t)
        attn_input = a_mat @ self.x  # shape => (N,)

        # plus input from W_in * u(t)
        input_term = self.W_in @ u   # shape => (N,)

        pre_activation = attn_input + input_term
        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.x = (1.0 - alpha)*self.x + alpha*x_new

    def collect_states(self, inputs, discard=100):
        """
        Teacher forcing => real input each step => dynamic adjacency => reservoir states
        """
        self.reset_state()
        states = []
        for val in inputs:
            self._update(val)
            states.append(self.x.copy())
        states = np.array(states)
        # shape => [T, reservoir_size]
        return states[discard:], states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        states_use, _ = self.collect_states(train_input, discard=discard)
        target_use = train_target[discard:]
        # augment with bias
        # X_aug = np.hstack([states_use, np.ones((states_use.shape[0],1))])

        # polynomial readout
        X_list = []
        for s in states_use:
            X_list.append(augment_state_with_squares(s))
        X_aug = np.array(X_list)  # shape => [T-discard, 2N+1]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, target_use)
        self.W_out = reg.coef_

    def predict_autoregressive(self, initial_input, n_steps):
        preds = []
        #self.reset_state()
        current_in = np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            # x_aug = np.concatenate([self.x, [1.0]])
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
            current_in = out
        return np.array(preds)
        
    def predict_open_loop(self, test_input):
        preds = []
        for true_input in test_input:
            self._update(true_input)
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)

