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

class MicrocolumnRes3D:
    """
    Cortical Microcolumn Reservoir for 3D chaotic systems.
    We have M columns, each with n_E excit + n_I inhib => total dimension N=M*(n_E+n_I).
    We'll define local E->E, E->I, I->E, I->I patterns plus some random inter-column edges.
    Then row-normalize + scale => final adjacency. 
    Standard leaky ESN update + polynomial readout => next-step 3D. 
    """

    def __init__(self,
                 n_columns=50,
                 n_excit=4,
                 n_inhib=1,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 prob_inter=0.02,     # fraction of inter-column edges
                 seed=42):
        """
        Parameters
        ----------
        n_columns      : M, number of microcolumns
        n_excit        : n_E, excit units per column
        n_inhib        : n_I, inhib units per column
        spectral_radius: final adjacency scale
        input_scale    : scale factor for W_in
        leaking_rate   : ESN leaky alpha
        ridge_alpha    : readout ridge penalty
        prob_inter     : fraction of inter-column edges that exist
        seed           : random seed
        """
        self.n_columns      = n_columns
        self.n_excit        = n_excit
        self.n_inhib        = n_inhib
        self.spectral_radius= spectral_radius
        self.input_scale    = input_scale
        self.leaking_rate   = leaking_rate
        self.ridge_alpha    = ridge_alpha
        self.prob_inter     = prob_inter
        self.seed           = seed

        self.N = n_columns*(n_excit+n_inhib)

        self.W     = None  # final shaped adjacency (N, N)
        self.W_in  = None  # input matrix (N, 3)
        self.W_out = None  # readout matrix (3, 2N+1)
        self.x     = None  # reservoir state (N,)

    def _build_microcolumn_adjacency(self):
        """
        Build adjacency with M columns. Each column => shape((n_E+n_I),(n_E+n_I)).
        We'll define local E->E positive, E->I positive (stronger?), I->E negative, I->I small or zero,.
        Then define random inter-column edges with small prob_inter. 
        Return W => shape(N,N).
        """
        np.random.seed(self.seed)
        # define block adjacency
        Ncol= self.n_excit + self.n_inhib # dimension per column
        M= self.n_columns
        N= self.N
        W= np.zeros((N,N))

        # We'll define local blocks
        wEE= 0.2
        wEI= 0.4
        wIE= -0.5
        wII= 0.0

        for c in range(M):
            off_r= c*Ncol
            # local random variation
            for e1 in range(self.n_excit):
                for e2 in range(self.n_excit):
                    if e1 != e2:
                        W[off_r+ e1, off_r+ e2]= wEE*(np.random.rand()*0.5+0.75)
                # E->I
                for i in range(self.n_inhib):
                    W[off_r+ e1, off_r+ self.n_excit+ i]= wEI*(np.random.rand()*0.5+0.75)
            for i1 in range(self.n_inhib):
                # I->E
                for e2 in range(self.n_excit):
                    W[off_r+ self.n_excit+ i1, off_r+ e2]= wIE*(np.random.rand()*0.5+0.75)
                # I->I
                for i2 in range(self.n_inhib):
                    if i1 != i2:
                        W[off_r+ self.n_excit+ i1, off_r+ self.n_excit+ i2]= wII

        # Now define inter-column edges with prob_inter => random excit or inhib => keep sign
        for c1 in range(M):
            off1= c1*Ncol
            for c2 in range(M):
                if c1 == c2: 
                    continue
                off2= c2*Ncol
                for i in range(Ncol):
                    for j in range(Ncol):
                        if np.random.rand() < self.prob_inter:
                            # sign depends on if i is excit or inhib
                            if i < self.n_excit:
                                sign = +1.0
                            else:
                                sign = -1.0
                            # magnitude random ~ 0.2
                            W[off1+ i, off2+ j] = sign*(0.2*np.random.rand())

        return W

    def _coactivation_shaping(self, data_3d, W_raw, W_in_shaping):
        """
        Shape W_raw from data coactivation while USING external input.
        We'll do a single pass:

        For consecutive t=0..(T-2):
            1) x(t+1) = tanh( W_raw x(t) + W_in_shaping u(t) ).
            2) coact_{ij} += x_i(t) * x_j(t+1).

        Finally we return coact as the shaped adjacency.  row-normalizing and spectral scaling it later.
        """
        T = data_3d.shape[0]   # number of time steps
        N = W_raw.shape[0]     # internal reservoir dimension
        coact = np.zeros((N, N))
        # Initialize reservoir state
        x = np.zeros(N)

        for t in range(T - 1):
            u_t = data_3d[t]  # shape (3,)
            x_next = np.tanh(W_raw @ x + W_in_shaping @ u_t)
            # Accumulate co-activations
            coact += np.outer(x, x_next)
            # Advance
            x = x_next

        return coact

    def fit_readout(self, train_input, train_target, discard=100):
        """
        1) build microcolumn adjacency => W_raw
        2) define a random W_in_shaping => shape(N,3), do coactivation shaping => W_shaped
        3) row-normalize + scale => self.W
        4) define final W_in => shape(N,3)
        5) teacher forcing => polynomial readout => self.W_out
        """
        # 1) build adjacency
        W_raw = self._build_microcolumn_adjacency()

        # 2) define a random W_in for shaping pass
        np.random.seed(self.seed + 50)
        W_in_shaping = (np.random.rand(self.N, 3) - 0.5)*2.0*self.input_scale

        # Perform coactivation shaping using external input
        W_shaped = self._coactivation_shaping(train_input, W_raw, W_in_shaping)

        # row normalize
        row_sum = W_shaped.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        W_shaped /= row_sum

        # scale spectral radius
        W_shaped = scale_spectral_radius(W_shaped, self.spectral_radius)
        self.W = W_shaped

        # 4) define final W_in => shape(N,3)
        np.random.seed(self.seed + 100)
        self.W_in = (np.random.rand(self.N, 3) - 0.5)*2.0*self.input_scale

        # 5) collect states => teacher forcing => polynomial => readout
        self.x = np.zeros(self.N)
        states_use, _ = self.collect_states(train_input, discard=discard)
        target_use = train_target[discard:]
        X_list= []
        for s in states_use:
            X_list.append(augment_state_with_squares(s))
        X_aug= np.array(X_list)

        reg= Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, target_use)
        self.W_out= reg.coef_

    def collect_states(self, inputs, discard=100):
        """
        Runs the reservoir on the given inputs (teacher-forcing the input only, no output feedback).
        Returns the reservoir states after discarding an initial transient.
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
        Standard leaky update:
          x_new = tanh(W @ x + W_in @ u)
          x <- (1-alpha)*x + alpha*x_new
        """
        alpha= self.leaking_rate
        pre_acts= self.W @ self.x + self.W_in @ u
        x_new= np.tanh(pre_acts)
        self.x= (1.0 - alpha)*self.x + alpha*x_new

    def predict_autoregressive(self, initial_input, n_steps):
        """
        Use the trained readout in closed loop for n_steps,
        starting from initial_input (3,).
        """
        preds= []
        current_in= np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            big_x= augment_state_with_squares(self.x)
            out= self.W_out @ big_x
            preds.append(out)
            current_in= out
        return np.array(preds)
    
    def predict_open_loop(self, test_input):
        preds = []
        for true_input in test_input:
            self._update(true_input)
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)