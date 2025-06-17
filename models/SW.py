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

class SW3D:
    """
    Cycle (ring) reservoir for 3D->3D single-step,
    teacher forcing for training, autoregressive for testing.
    """
    def __init__(self,
                 reservoir_size=300,
                 edges_per_node=6,                              #E in paper
                 input_reservoir_size=100,                      #Nin
                 output_reservoir_size=100,                     #Nout
                 rewiring_probability=0.1,                      #p in paper
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=48):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.edges_per_node = edges_per_node
        self.input_res_size = input_reservoir_size
        self.output_res_size = output_reservoir_size
        self.rewiring_probability = rewiring_probability

        np.random.seed(self.seed+1)
        self.W_in = (np.random.rand(reservoir_size,3) - 0.5)*self.input_scale

        self.W_out = None
        self.x = np.zeros(reservoir_size)

        W = np.zeros((self.reservoir_size, self.reservoir_size))
        for i in range(self.reservoir_size):
            for offset in range(1, 4):
                W[i, (i + offset) % self.reservoir_size] = 1.0
                W[i, (i - offset) % self.reservoir_size] = 1.0

        for i in range(self.reservoir_size):
            current_connections = np.where(W[i] == 1.0)[0]
            for j in current_connections:
                if np.random.rand() < self.rewiring_probability:
                    W[i, j] = 0.0
                    possible_nodes = list(set(range(self.reservoir_size)) - {i} - set(np.where(W[i] == 1.0)[0]))
                    if possible_nodes:
                        new_j = np.random.choice(possible_nodes)
                        W[i, new_j] = 1.0

        W = scale_spectral_radius(W, self.spectral_radius)
        self.W = W

    # def reset_state(self):
    #     self.x = np.zeros(self.reservoir_size)

    # def _update(self, u):
    #     pre_activation = self.W @ self.x + self.W_in @ u
    #     x_new = np.tanh(pre_activation)
    #     alpha = self.leaking_rate
    #     self.x = (1.0 - alpha)*self.x + alpha*x_new

    # def collect_states(self, inputs, discard=100):
    #     self.reset_state()
    #     states = []
    #     for val in inputs:
    #         self._update(val)
    #         states.append(self.x.copy())
    #     states = np.array(states)
    #     return states[discard:], states[:discard]

    # def fit_readout(self, train_input, train_target, discard=100):
    #     states_use, _ = self.collect_states(train_input, discard=discard)
    #     targets_use = train_target[discard:]
    #     # X_aug = np.hstack([states_use, np.ones((states_use.shape[0],1))])

    #     # polynomial readout
    #     X_list = []
    #     for s in states_use:
    #         X_list.append(augment_state_with_squares(s))
    #     X_aug = np.array(X_list)  # shape => [T-discard, 2N+1]

    #     reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
    #     reg.fit(X_aug, targets_use)
    #     self.W_out = reg.coef_

    # def predict_autoregressive(self, initial_input, n_steps):
    #     preds = []
    #     current_in = np.array(initial_input)
    #     for _ in range(n_steps):
    #         self._update(current_in)
    #         # x_aug = np.concatenate([self.x, [1.0]])
    #         x_aug = augment_state_with_squares(self.x)
    #         out = self.W_out @ x_aug
    #         preds.append(out)
    #         current_in = out
    #     return np.array(preds)
    
# --- Revised segregated IO reservoir class ---
class SW3DSegregated:
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
        
        np.random.seed(self.seed+1)
        
        # --- Assign spatial positions to nodes (angular embedding on a circle) ---
        # Define width for the input/output clusters.
        delta = np.pi / 20  # small angular window (≈9°)
        n_in = self.input_res_size
        n_out = self.output_res_size
        n_core = self.core_res_size
        
        # Input nodes: angles uniformly in [-delta, delta]
        input_positions = np.random.uniform(-delta, delta, size=n_in)
        # Output nodes: angles uniformly in [pi-delta, pi+delta]
        output_positions = np.random.uniform(np.pi - delta, np.pi + delta, size=n_out)
        
        # Core nodes: distribute them evenly over the remaining parts of the circle.
        # For clarity, we divide the remainder into two arcs.
        n_core1 = n_core // 2
        n_core2 = n_core - n_core1
        # Arc 1: from delta to (pi - delta)
        core_positions1 = np.linspace(delta, np.pi - delta, n_core1, endpoint=False)
        # Arc 2: from (pi + delta) to (2*pi - delta)
        core_positions2 = np.linspace(np.pi + delta, 2*np.pi - delta, n_core2, endpoint=False)
        core_positions = np.concatenate([core_positions1, core_positions2])
        
        # Concatenate all positions and assign labels:
        # 0 for input; 1 for core; 2 for output.
        self.positions = np.concatenate([input_positions, core_positions, output_positions])
        self.labels = np.concatenate([np.zeros(n_in, dtype=int),
                                       np.ones(n_core, dtype=int),
                                       2*np.ones(n_out, dtype=int)])
        
        # Sort nodes by angular position so they lie along a ring.
        sort_order = np.argsort(self.positions)
        self.positions = self.positions[sort_order]
        self.labels = self.labels[sort_order]
        self.all_indices = np.arange(self.reservoir_size)
        
        # Determine indices for each group after sorting.
        self.input_indices = np.where(self.labels == 0)[0]
        self.core_indices = np.where(self.labels == 1)[0]
        self.output_indices = np.where(self.labels == 2)[0]
        
        # --- Create input weight matrix (applied only to input nodes) ---
        # Assuming external input is 3-dimensional.
        self.W_in = (np.random.rand(len(self.input_indices), 3) - 0.5) * self.input_scale
        
        # Initialize output weight (readout) and reservoir state vector.
        self.W_out = None
        self.x = np.zeros(self.reservoir_size)
        
        # --- Build reservoir connectivity using the sorted (ring) order ---
        # We use a simple ring-based connection: for each node, connect to its
        # nearest neighbors (with offsets 1 to edges_per_node/2) in each direction.
        W = np.zeros((self.reservoir_size, self.reservoir_size))
        half_edges = self.edges_per_node // 2
        for i in range(self.reservoir_size):
            for offset in range(1, half_edges + 1):
                j = (i + offset) % self.reservoir_size
                k = (i - offset) % self.reservoir_size
                W[i, j] = 1.0
                W[i, k] = 1.0
        
        # --- Apply random rewiring (Watts–Strogatz) across all nodes ---
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
                        
        # Scale the reservoir connectivity to have the desired spectral radius.
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
        ext_input = np.zeros(self.reservoir_size)
        # Apply external input only to nodes in the input group.
        for idx, node in enumerate(self.input_indices):
            ext_input[node] = self.W_in[idx] @ u
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
    
    def predict_open(self, test_input):
        preds = []
        for u in test_input:
            self._update(u)
            x_out = self.x[self.output_indices]
            x_aug = augment_state_with_squares(x_out)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)


# --- Random IO Reservoir Class ---
class SW3DRandom:
    """
    Random IO small-world reservoir for 3D->3D mapping.
    
    In this version:
      - Total reservoir nodes: reservoir_size (default 500)
      - Input nodes: a subset of reservoir nodes (default 100) chosen at random.
      - Output nodes: a disjoint subset of reservoir nodes (default 100) chosen at random.
      - The remaining nodes (core nodes) are the other 300 nodes.
    
    The recurrent connectivity is created using a ring-based nearest-neighbor scheme (with edges_per_node
    number of connections per node) and then rewired randomly with a given rewiring probability.
    
    The external input (assumed 3-dimensional) is injected only on the input nodes (via W_in),
    and the output is read only from the output nodes.
    """
    def __init__(self,
                 reservoir_size=500,
                 edges_per_node=6,              # Must be even (here: 3 neighbors forward, 3 backward)
                 input_reservoir_size=100,        # Number of input nodes
                 output_reservoir_size=100,       # Number of output nodes
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

        np.random.seed(self.seed+1)
        total_indices = np.arange(self.reservoir_size)
        # Randomly choose indices for IO nodes without overlapping.
        all_io = np.random.choice(total_indices, size=(self.input_res_size + self.output_res_size), replace=False)
        np.random.shuffle(all_io)
        self.input_indices = all_io[:self.input_res_size]
        self.output_indices = all_io[self.input_res_size:self.input_res_size + self.output_res_size]
        # The core indices are the remaining ones
        self.core_indices = np.array(list(set(total_indices) - set(self.input_indices) - set(self.output_indices)))
        
        # --- Create Input Weight Matrix ---
        # Only input nodes receive external input; input is assumed 3-dimensional.
        self.W_in = (np.random.rand(len(self.input_indices), 3) - 0.5) * self.input_scale

        # Initialize output weight (readout) and reservoir state vector.
        self.W_out = None
        self.x = np.zeros(self.reservoir_size)
        
        # --- Build reservoir connectivity ---
        # Use a ring-based connectivity in the natural (index) order.
        # Each node i will be connected to its nearest half_edges neighbors in both directions.
        W = np.zeros((self.reservoir_size, self.reservoir_size))
        half_edges = self.edges_per_node // 2
        for i in range(self.reservoir_size):
            for offset in range(1, half_edges + 1):
                j = (i + offset) % self.reservoir_size
                k = (i - offset) % self.reservoir_size
                W[i, j] = 1.0
                W[i, k] = 1.0

        # --- Random rewiring (Watts-Strogatz style) ---
        for i in range(self.reservoir_size):
            current_neighbors = np.where(W[i] == 1.0)[0]
            for j in current_neighbors:
                if np.random.rand() < self.rewiring_probability:
                    W[i, j] = 0.0
                    possible_nodes = list(set(range(self.reservoir_size)) - {i} - set(np.where(W[i] == 1.0)[0]))
                    if possible_nodes:
                        new_j = np.random.choice(possible_nodes)
                        W[i, new_j] = 1.0

        # Scale the reservoir weight matrix to have the desired spectral radius.
        self.W = scale_spectral_radius(W, self.spectral_radius)
    
    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)
    
    def _update(self, u):
        """
        Update the reservoir state.
          - The recurrent input is W @ x.
          - External input u (3D) is added only at input nodes.
          - A tanh nonlinearity and leaky integration update the state.
        """
        recurrent = self.W @ self.x
        ext_input = np.zeros(self.reservoir_size)
        for idx, node in enumerate(self.input_indices):
            ext_input[node] = self.W_in[idx] @ u
        pre_activation = recurrent + ext_input
        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.x = (1.0 - alpha) * self.x + alpha * x_new
    
    def collect_states(self, inputs, discard=100):
        """
        Drive the reservoir with a sequence of inputs and record the states.
        Only the states of the output nodes are kept for the readout.
        """
        self.reset_state()
        all_states = []
        for u in inputs:
            self._update(u)
            all_states.append(self.x.copy())
        all_states = np.array(all_states)
        output_states = all_states[:, self.output_indices]
        return output_states[discard:], all_states[:discard]
    
    def fit_readout(self, train_input, train_target, discard=100):
        """
        Train the readout weights from the reservoir.
          - The readout is based on states from output nodes.
          - Each state is augmented (with squared values and bias) before Ridge regression.
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]
        X_list = [augment_state_with_squares(s) for s in states_use]
        X_aug = np.array(X_list)
        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, targets_use)
        self.W_out = reg.coef_
    
    def predict_autoregressive(self, initial_input, n_steps):
        """
        Generate predictions in an autoregressive manner.
          - At each step, the reservoir is updated with the current input.
          - The output is computed solely from the output nodes.
          - The predicted output is used as the next input.
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
    
    def predict_open(self, test_input):
        preds = []
        for u in test_input:
            self._update(u)
            x_out = self.x[self.output_indices]
            x_aug = augment_state_with_squares(x_out)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)