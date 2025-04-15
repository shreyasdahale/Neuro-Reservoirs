import numpy as np
from sklearn.linear_model import Ridge

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

class HFRRes3D:
    """
    Hierarchical Fractal Reservoir (HFR) for 3D chaotic systems.
    
    This novel reservoir architecture partitions the chaotic attractor at multiple
    hierarchical scales, combining them in a fractal-like adjacency structure.
    The method is model-free, relying solely on the observed trajectory in R^3,
    and does not require knowledge of any system parameters such as sigma, rho, beta
    for Lorenz63. 
    
    Key Idea:
     1) Define multiple 'scales' of partition of the data's bounding region.
     2) Each scale is subdivided into a certain number of cells (regions).
     3) Each cell at level l has links to both:
        - other cells at the same level (horizontal adjacency),
        - 'child' cells at the finer level l+1 (vertical adjacency).
     4) We gather all cells across levels => a multi-level fractal graph => adjacency => W.
     5) We build a typical ESN from this adjacency, feed data with W_in, run leaky tanh updates,
        then do a polynomial readout for 3D next-step prediction.

    This approach is suitable for chaotic systems whose attractors often exhibit fractal
    self-similarity, thus capturing multi-scale structures in a single reservoir.
    """

    def _init_(self,
                 n_levels=3,             # number of hierarchical levels
                 cells_per_level=None,   # list of number of cells at each level, e.g. [8, 32, 128]
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):
        """
        Parameters
        ----------
        n_levels       : int, number of hierarchical scales
        cells_per_level: list[int], the number of partitions/cells at each level
                         if None, we auto-generate e.g. 2^(level+2)
        spectral_radius: final scaling for adjacency
        input_scale    : random input scale W_in
        leaking_rate   : ESN leaky alpha
        ridge_alpha    : readout ridge penalty
        seed           : random seed
        """
        self.n_levels        = n_levels
        self.cells_per_level = cells_per_level
        self.spectral_radius = spectral_radius
        self.input_scale     = input_scale
        self.leaking_rate    = leaking_rate
        self.ridge_alpha     = ridge_alpha
        self.seed            = seed

        if self.cells_per_level is None:
            # default scheme e.g. 8, 16, 32 for 3 levels
            self.cells_per_level = [8*(2**i) for i in range(n_levels)]

        # We'll store adjacency W, input W_in, readout W_out, reservoir state x
        self.W     = None
        self.W_in  = None
        self.W_out = None
        self.x     = None

        # We'll define a total number of nodes = sum(cells_per_level)
        self.n_nodes = sum(self.cells_per_level)

    def _build_partitions(self, data_3d):
        """
        Build hierarchical partitions for each level.
        We'll store the bounding box for data_3d, then for each level l in [0..n_levels-1]
        run e.g. k-means with K = cells_per_level[l], each point gets a label => we track transitions.

        Return: 
          partitions => list of arrays, partitions[l] => shape (N, ) cluster assignment in [0..cells_per_level[l]-1]
        """
        from sklearn.cluster import KMeans
        N = len(data_3d)
        partitions = []

        for level in range(self.n_levels):
            k = self.cells_per_level[level]
            # cluster
            kmeans = KMeans(n_clusters=k, random_state=self.seed+10*level)
            kmeans.fit(data_3d)
            labels = kmeans.predict(data_3d)
            partitions.append(labels)

        return partitions

    def _build_hierarchical_adjacency(self, data_3d):
        """
        Build a block adjacency with cross-level links, then scale spectral radius.
        Steps:
          1) Build partitions for each level => partitions[l] in [0..cells_per_level[l]-1]
          2) For each level l, build a transition matrix T_l of shape (cells_per_level[l], cells_per_level[l]).
          3) Link scale l to scale l+1 by figuring out which cluster i at scale l maps to which cluster j at scale l+1
             for each sample t => link i-> j if data_3d[t] is in i at scale l and j at scale l+1.
          4) Combine all transitions in one big adjacency W in R^(n_nodes x n_nodes).
          5) row-normalize W => scale largest eigenvalue => spectral_radius
        """
        partitions = self._build_partitions(data_3d)
        N = len(data_3d)

        # offsets for each level => to index big W
        offsets = []
        running = 0
        for level in range(self.n_levels):
            offsets.append(running)
            running += self.cells_per_level[level]

        # total nodes
        n_tot = self.n_nodes
        # initialize adjacency
        A = np.zeros((n_tot, n_tot))

        # 1) horizontal adjacency in each level
        for level in range(self.n_levels):
            k = self.cells_per_level[level]
            labels = partitions[level]
            # T_l => shape (k, k)
            T_l = np.zeros((k, k))
            for t in range(N-1):
                i = labels[t]
                j = labels[t+1]
                T_l[i,j]+=1
            # row normalize
            row_sum = T_l.sum(axis=1, keepdims=True)
            row_sum[row_sum==0.0] = 1.0
            T_l /= row_sum
            # place T_l into big A
            off = offsets[level]
            A[off:off+k, off:off+k] = T_l

        # 2) vertical adjacency between scale l and l+1
        for level in range(self.n_levels-1):
            k_l   = self.cells_per_level[level]
            k_lp1 = self.cells_per_level[level+1]
            labels_l   = partitions[level]
            labels_lp1 = partitions[level+1]
            # we define adjacency from i in [0..k_l-1] to j in [0..k_lp1-1] if the same sample t belongs to i at level l and j at l+1
            # Count how many times
            Xvert = np.zeros((k_l, k_lp1))
            for t in range(N):
                i = labels_l[t]
                j = labels_lp1[t]
                Xvert[i,j]+=1
            # row normalize
            row_sum = Xvert.sum(axis=1, keepdims=True)
            row_sum[row_sum==0.0] = 1.0
            Xvert /= row_sum
            # place in big A
            off_l   = offsets[level]
            off_lp1 = offsets[level+1]
            A[off_l:off_l+k_l, off_lp1:off_lp1+k_lp1] = Xvert
            # tentative idea, we could also define adjacency from l+1 -> l (parent link), if desired
            # we do the same for the 'child -> parent' link or skip it if we only want forward adjacency
            # For now, let's do symmetrical
            Yvert = Xvert.T
            col_sum = Yvert.sum(axis=1, keepdims=True)
            col_sum[col_sum==0.0] = 1.0
            Yvert /= col_sum
            A[off_lp1:off_lp1+k_lp1, off_l:off_l+k_l] = Yvert

        # now we have a big adjacency => row normalize again, then scale spectral radius
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum==0.0] = 1.0
        A /= row_sum

        A = scale_spectral_radius(A, self.spectral_radius)
        return A

    def fit_readout(self, train_input, train_target, discard=100):
        """
        Main training routine:
          1) Build hierarchical adjacency from fractal partition => self.W
          2) define W_in => shape(n_nodes, 3)
          3) teacher forcing => polynomial readout => solve => self.W_out
        """
        np.random.seed(self.seed)
        # Build adjacency
        W_big = self._build_hierarchical_adjacency(train_input)
        self.W = W_big

        # define W_in => shape(n_nodes,3)
        self.n_nodes = W_big.shape[0]
        self.W_in = (np.random.rand(self.n_nodes,3)-0.5)*2.0*self.input_scale

        # define reservoir state
        self.x = np.zeros(self.n_nodes)

        # gather states => teacher forcing => polynomial => readout
        states_use, _ = self.collect_states(train_input, discard=discard)
        target_use = train_target[discard:]
        X_list= []
        for s in states_use:
            X_list.append( augment_state_with_squares(s) )
        X_aug= np.array(X_list)

        reg= Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, target_use)
        self.W_out= reg.coef_

    def collect_states(self, inputs, discard=100):
        """
        Teacher forcing => feed real 3D => gather states => shape => [T-discard, n_nodes].
        returns (states_after_discard, states_discarded).
        """
        self.reset_state()
        states= []
        for val in inputs:
            self._update(val)
            states.append(self.x.copy())
        states= np.array(states)
        return states[discard:], states[:discard]

    def reset_state(self):
        if self.x is not None:
            self.x.fill(0.0)

    def _update(self, u):
        """
        x(t+1)= (1-alpha)x(t)+ alpha tanh( W*x(t)+ W_in*u(t) ).
        """
        alpha= self.leaking_rate
        pre_acts= self.W@self.x + self.W_in@u
        x_new= np.tanh(pre_acts)
        self.x= (1.0- alpha)*self.x+ alpha*x_new

    def predict_autoregressive(self, initial_input, n_steps):
        """
        fully autonomous => feed last predicted => next input
        """
        preds= []
        #self.reset_state()
        current_in= np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            big_x= augment_state_with_squares(self.x)
            out= self.W_out@big_x
            preds.append(out)
            current_in= out
        return np.array(preds)