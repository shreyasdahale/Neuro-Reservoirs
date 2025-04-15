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

class SW3DSegregated2:
    """
    Segregated IO small-world reservoir with spatial embedding.
    
    The reservoir is partitioned into three segments:
      - Input group: nodes that receive external input.
      - Core group: intermediate hidden nodes.
      - Output group: nodes from which the readout is taken.
    
    Total nodes = 500:
      - 100 Input nodes are clustered (angles in [-delta, delta])
      - 100 Output nodes are clustered (angles in [pi-delta, pi+delta])
      - 300 Core nodes are distributed evenly over the remaining circle.
    
    Connectivity is generated over the sorted order (by angular position) using
    a regular ring (nearest neighbor) scheme with random rewiring (Watts–Strogatz style).
    """
    def __init__(self,
                 reservoir_size=500,
                 edges_per_node=6,            # Must be even (here 6: 3 forward, 3 backward)
                 input_reservoir_size=100,      # Number of input nodes
                 output_reservoir_size=100,     # Number of output nodes
                 rewiring_probability=0.1,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=48):
        # Save parameters
        self.reservoir_size = reservoir_size
        self.edges_per_node = edges_per_node
        self.input_res_size = input_reservoir_size
        self.output_res_size = output_reservoir_size
        self.core_res_size = reservoir_size - (input_reservoir_size + output_reservoir_size)
        self.rewiring_probability = rewiring_probability
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.all_indices = np.arange(self.reservoir_size)

        core_res_size=self.reservoir_size - self.input_res_size -self.output_res_size

        self.input_indices = np.arange(0,self.input_res_size)
        self.core_indices = np.concatenate([np.arange(self.input_res_size, self.input_res_size + core_res_size/2),
                         np.arange(self.input_res_size+core_res_size/2 +self.output_res_size, self.reservoir_size)])
        self.output_indices = np.arange(self.input_res_size + core_res_size/2,self.input_res_size+core_res_size/2 +self.output_res_size )

        self.W_in = (np.random.rand(len(self.input_indices), 3) - 0.5) * self.input_scale

        self.W_out = None
        self.x = np.zeros(self.reservoir_size)

        W = np.zeros((self.reservoir_size, self.reservoir_size))
        half_edges = self.edges_per_node // 2
        for i in range(self.reservoir_size):
            for offset in range(1, half_edges + 1):
                j = (i + offset) % self.reservoir_size
                k = (i - offset) % self.reservoir_size
                W[i, j] = 1.0
                W[i, k] = 1.0

        for i in range(self.reservoir_size):
            current_neighbors = np.where(W[i] == 1.0)[0]
            for j in current_neighbors:
                if np.random.rand() < self.rewiring_probability:
                    W[i, j] = 0.0
                    # Ensure new connection does not duplicate an existing link
                    possible_nodes = list(set(self.all_indices) - {i} - set(np.where(W[i] == 1.0)[0]))
                    if possible_nodes:
                        new_j = np.random.choice(possible_nodes)
                        W[i, new_j] = 1.0

        self.W = scale_spectral_radius(W, self.spectral_radius)

    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)

    def _update(self, u):
        """
        Update the reservoir state vector x.
          - The recurrent contribution is computed via W @ x.
          - The external input u (a 3D vector) is added only to input nodes (via W_in).
          - A tanh nonlinearity and leaky integration are applied.
        """
        recurrent = self.W @ self.x
        #The next line only works coz input nodes are 0 to 100
        ext_input = np.concatenate([self.W_in @ u, np.zeros(self.reservoir_size-self.input_res_size)]) 

        pre_activation = recurrent + ext_input
        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.x = (1.0 - alpha) * self.x + alpha * x_new

    def collect_states(self, inputs, discard=100):
        """
        Run the reservoir over an input sequence.
          - The external input drives only the input nodes.
          - States from the output nodes are recorded (after a washout period).
        Returns:
          - output_states: a T x (# output nodes) array (post-discard).
          - discard_states: states corresponding to the washout period.
        """
        self.reset_state()
        all_states = []
        for u in inputs:
            self._update(u)
            all_states.append(self.x.copy())
        all_states = np.array(all_states)
        output_states = all_states[:, self.output_indices]
        return output_states[discard:], output_states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        """
        Train the linear readout on teacher-forced reservoir states.
        Only states from the output nodes (the segregated readout cluster)
        are used. A polynomial (squares) expansion and a bias term are
        applied before ridge regression.
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]
        X_list = [augment_state_with_squares(s) for s in states_use]
        X_aug = np.array(X_list)  # Shape: [T-discard, 2*len(output_indices) + 1]
        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, targets_use)
        self.W_out = reg.coef_

    def predict_autoregressive(self, initial_input, n_steps):
        """
        Generate predictions in closed-loop (autoregressive) mode.
          - At each step, update the reservoir.
          - Compute the readout using only the output node states.
          - The prediction becomes the next input.
        """
        preds = []
        current_in = np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            x_out = self.x[self.output_indices]
            x_aug = augment_state_with_squares(x_out)
            out = self.W_out @ x_aug
            preds.append(out)
            current_in = out
        return np.array(preds)


