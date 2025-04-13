
import numpy as np
print("np:", np)
print("type(np):", type(np))
print("np.zeros:", getattr(np, "zeros", "missing"))


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

class smallWorld:
    def __init__(
        self,
        reservoir_size=500,
        edges_per_node=6,                               #E in paper
        input_reservoir_size=100,                       #Nin
        output_reservoir_size=100,                        #Nout
        rewiring_probability=0.1,                       #p in paper
        spectral_radius=0.95,                           #alpha
        regularization_parameter=pow(10,-10),
        seed=47
    ):
        self.reservoir_size = reservoir_size
        self.edges_per_node = edges_per_node
        self.input_res_size = input_reservoir_size
        self.output_res_size = output_reservoir_size
        self.rewiring_probability = rewiring_probability
        self.spectral_radius = spectral_radius
        self.regularization_parameter = regularization_parameter
        self.seed = seed

        np.random.seed(self.seed)

        self.build_small_world()

    def build_small_world(self):
        
        N = int(self.reservoir_size)
        Nin = int(self.input_res_size)
        E = int(self.edges_per_node)

        print(type(np))

        #Initializing weights for input nodes
        W_in_nodes = np.random.rand(Nin,3) - 0.5

        #Concatenating with zero matrix to get appropriate dimension
        self.W_in = np.vstack((W_in_nodes, np.zeros((N-Nin,3))))

        #Generating regular connection
        regular = np.zeros((N , N), dtype =float)

        for i in range(0,N):
            for j in range(1,E//2):
                regular[i][(i-j) % N] = 1
                regular[i][(i+j) % N] = 1
        
        #Rewiring 
        p = self.rewiring_probability
        rewired = np.zeros((N,N))
        
        for i in range(0,N):
            for j in range(-E//2, E//2 + 1):
                if regular[i][(i+j)%N] == 1 :
                    random_num = np.random.rand(1)
                    if random_num <= p:
                        n = np.random.randint(i+E/2+1,i+N-E/2)  
                        # basically this n mod N will give number of a node other than the 
                        # neighbours it is connected to
                        rewired[i][n%N] = 1
                    else:
                        rewired[i][(i+j)%N] = 1 
                        #just making the unchanged edge conection from regular in the case of not getting rewired

        #scaling so that edge weights are in uniform distribution of -0.5 to 0.5
        #So then we get initial weight matrix Wo

        scaling_matrix = np.random.rand(N,N) - 0.5
        Wo = rewired * scaling_matrix
        
        #Scaling to get spectral radius equal to alpha
        self.W = scale_spectral_radius(Wo , self.spectral_radius)

        self.W_out = None
        self.x = np.zeros(N)

    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)

    def update(self,u):
        pre_activation = self.W_in @ u + self.W @ self.x
        self.x = np.tanh(pre_activation)

    #collecting states of full reservoir
    def collect_states(self, inputs, discard=100):
        self.reset_state()
        states=[]
        for val in inputs:
            self.update(val)
            states.append(self.x.copy())
        states = np.array(states)
        return states
    
    #function to get x_out states because readout only done from output nodes
    def collect_states_output(self, inputs, discard=100):
        states = self.collect_states(inputs, discard=discard)
        
        Nin = self.input_res_size
        N = self.reservoir_size
        Nout = self.output_res_size
        #since it is segregated, and the output is on exactly opposite side of input in the ring
        #therefore if input is from 0 to 100 then output has to be 250 to 350 so it is not exact opposite side
        #So this can be written as 
        # Nin/2(midpoint of input nodes) + N/2(exact opposite to input midpoint) - Nout/2(half output nodes before this midpoint) to 
        # Nin/2 + N/2 + Nout/2
        output_states = states[:,Nin//2 + N//2 - Nout//2 : Nin//2 + N//2 + Nout//2]       
        return output_states[discard:], output_states[:discard]
    
    def fit_readout(self, train_input, train_target, discard=100):
        states_use, _ =self.collect_states_output(train_input, discard=discard)
        targets_use =train_target[discard:]
        X_aug = np.hstack([states_use, np.ones((states_use.shape[0],1))])
        S = X_aug
        D = targets_use
        beta = self.regularization_parameter

        #Calculating Wout according to formula given in paper
        Wout = (np.linalg.inv(S.T @ S + beta * np.eye(self.output_res_size+1)) @ S.T @ D).T 
        self.W_out = Wout


    def predict_autoregressive(self, initial_input, n_steps):
        preds = []
        current_in = np.array(initial_input)
        Nin = self.input_res_size
        N = self.reservoir_size
        Nout = self.output_res_size
        for _ in range(n_steps):
            self.update(current_in)
            x_out = self.x[Nin//2 + N//2 - Nout//2 : Nin//2 + N//2 + Nout//2]
            x_aug = np.concatenate([x_out, [1.0]])
            out = self.W_out @ x_aug
            preds.append(out)
            current_in = out
        return np.array(preds)