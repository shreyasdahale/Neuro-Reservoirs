"""
Diffusion-Wavelet Multi-Scale Reservoir (DW-MSR)
===============================================

-----------------------------------------------------------------------
Notation recap
-----------------------------------------------------------------------
* Base graph         G = (V,E), |V| = n0
* Laplacian          L = D - A
* Diffusion kernel   P_s = exp(- 2**s · τ0 · L)      for s = 0 … S
* State vector       x_t = [x_t^{(0)} ; … ; x_t^{(S)}] ∈ R^{N}
                      where   N = (S+1) · n0
* Update (per scale) see equations in the methodology.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm
from sklearn.linear_model import Ridge
from typing import Sequence


# --------------------------------------------------------------------- #
#                          Helper functions                             #
# --------------------------------------------------------------------- #
def _build_laplacian(adj: sparse.spmatrix) -> sparse.spmatrix:
    """Combinatorial Laplacian L = D - A   (sparse CSR)."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    D = sparse.diags(deg, format="csr")
    return D - adj


def _default_sequence(val: float, length: int) -> list[float]:
    """Repeat *val* 'length' times, return as list."""
    return [val] * length


# --------------------------------------------------------------------- #
#                     Diffusion-Wavelet Reservoir ESN                   #
# --------------------------------------------------------------------- #
class DWMSR3D:
    """
    DW-MSR Echo-State Network.

    Parameters
    ----------
    adj                   : scipy.sparse matrix (shape [n0,n0])
        Symmetric, unweighted or weighted adjacency of the base graph.
    num_scales            : int,   S ≥ 0   (# coarse levels)
    tau0                  : float, base diffusion time   (τ₀)
    betas                 : Sequence[float] length S,   funnel strengths β_s
    alphas                : Sequence[float] length S+1, leak per scale α_s
    input_scale           : float, scale of random W_in entries
    ridge_alpha           : float, ℓ₂ penalty in ridge read-out
    detail_features       : bool,  include Δ_s = x^{(s-1)}-x^{(s)} in Φ(x)?

    Notes
    -----
    * Reservoir size  N = (S+1) * n0
    * P_s are pre-computed once with sparse expm; they share sparsity of *adj*.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        adj: sparse.spmatrix,
        num_scales: int = 2,
        tau0: float = 0.1,
        betas: Sequence[float] | None = None,
        alphas: Sequence[float] | None = None,
        input_scale: float = 0.5,
        ridge_alpha: float = 1e-6,
        detail_features: bool = True,
        seed: int = 42,
    ):
        # -------- basics -------------------------------------------------
        num_scales = len(betas) if betas is not None else num_scales
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("adjacency must be square")
        if not sparse.isspmatrix(adj):
            adj = sparse.csr_matrix(adj)
        self.n0 = adj.shape[0]
        self.S = int(num_scales)
        if self.S < 0:
            raise ValueError("num_scales must be ≥ 0")

        self.N = (self.S + 1) * self.n0
        self.tau0 = float(tau0)
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        self.detail_features = detail_features
        self.seed = seed

        # -------- leak & funnel parameters ------------------------------
        self.betas = list(betas) if betas is not None else _default_sequence(0.5, self.S)
        if len(self.betas) != self.S:
            raise ValueError("betas must have length S")

        self.alphas = (
            list(alphas)
            if alphas is not None
            else [0.5] + _default_sequence(1.0, self.S)  # finer quicker, coarse slow
        )
        if len(self.alphas) != self.S + 1:
            raise ValueError("alphas must have length S+1")

        # -------- internal matrices -------------------------------------
        self.Ps: list[sparse.spmatrix] = []
        self.Vs: list[np.ndarray] = []  # just β_s I, store scalars
        self._precompute_operators(adj)

        self.W_in: np.ndarray | None = None      # set in fit_readout
        self.W_out: np.ndarray | None = None

        # state block list for convenience (each block length n0)
        self.x_blocks = [np.zeros(self.n0, dtype=np.float32) for _ in range(self.S + 1)]

    # ------------------------------------------------------------------ #
    #                    Pre-computation of diffusion kernels            #
    # ------------------------------------------------------------------ #
    def _precompute_operators(self, adj: sparse.spmatrix):
        """Compute P_s and store funnel scalars β_s."""
        L = _build_laplacian(adj).tocsr()
        # largest eigenvalue bound (Gershgorin): max row sum of |L|
        lam_max = L.max(axis=1).toarray().ravel().max() + 1e-9
        if self.tau0 * (2 ** self.S) * lam_max > 50:
            print(
                "Warning: very large diffusion times may cause underflow in expm; "
                "consider reducing tau0."
            )

        for s in range(self.S + 1):
            tau_s = (2 ** s) * self.tau0
            Ps = expm((-tau_s) * L)  # still sparse CSR
            self.Ps.append(Ps)

        self.Vs = self.betas  # just scalars

    # ------------------------------------------------------------------ #
    #                            Core update                             #
    # ------------------------------------------------------------------ #
    def _single_step(self, u_t: np.ndarray):
        """
        Update all scales in causal order (fine → coarse) per eq. (1).
        """
        new_blocks = []

        # scale 0 (fine)
        z0 = self.Ps[0].dot(self.x_blocks[0]) + self.W_in.dot(u_t)
        x0_new = np.tanh(z0)
        x0_next = (1.0 - self.alphas[0]) * self.x_blocks[0] + self.alphas[0] * x0_new
        new_blocks.append(x0_next)

        # coarser scales
        for s in range(1, self.S + 1):
            z = self.Ps[s].dot(self.x_blocks[s]) + self.Vs[s - 1] * new_blocks[s - 1]
            xs_new = np.tanh(z)
            xs_next = (1.0 - self.alphas[s]) * self.x_blocks[s] + self.alphas[s] * xs_new
            new_blocks.append(xs_next)

        # commit
        self.x_blocks = new_blocks

    def reset_state(self):
        for blk in self.x_blocks:
            blk.fill(0.0)

    # ------------------------------------------------------------------ #
    #                        Read-out training                            #
    # ------------------------------------------------------------------ #
    def fit_readout(self, inputs: np.ndarray, targets: np.ndarray, discard: int = 100):
        """
        Teacher-forcing to train W_out (ridge).

        inputs  shape [T, d_in]
        targets shape [T, d_out]
        """
        T, d_in = inputs.shape
        if T <= discard + 1:
            raise ValueError("Not enough data for training")

        # random W_in
        rng = np.random.default_rng(self.seed)
        self.W_in = (
            rng.uniform(-1.0, 1.0, size=(self.n0, d_in)) * self.input_scale
        ).astype(np.float32)

        # roll through data
        self.reset_state()
        states, details = [], []
        for t in range(T):
            self._single_step(inputs[t])
            if t >= discard:
                flat_state = np.concatenate(self.x_blocks)
                states.append(flat_state)

                if self.detail_features and self.S > 0:
                    # Δ_s = x^{(s-1)} − x^{(s)}
                    delta_list = [
                        self.x_blocks[s - 1] - self.x_blocks[s] for s in range(1, self.S + 1)
                    ]
                    details.append(np.concatenate(delta_list))

        X_main = np.asarray(states, dtype=np.float32)  # [T-d, N]
        Y = targets[discard:]

        # feature map Φ
        if self.detail_features and self.S > 0:
            X_det = np.asarray(details, dtype=np.float32)  # same rows
            feats = np.concatenate(
                [X_main, X_det, np.ones((X_main.shape[0], 1), dtype=np.float32)], axis=1
            )
        else:
            feats = np.concatenate(
                [X_main, np.ones((X_main.shape[0], 1), dtype=np.float32)], axis=1
            )

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(feats, Y)
        self.W_out = reg.coef_.astype(np.float32)

    # ------------------------------------------------------------------ #
    #                       Autoregressive rollout                        #
    # ------------------------------------------------------------------ #
    def predict_autoregressive(
        self, initial_input: np.ndarray, n_steps: int
    ) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("fit_readout() must be called first")

        d_in = initial_input.shape[0]
        d_out = self.W_out.shape[0]
        preds = np.empty((n_steps, d_out), dtype=np.float32)

        #self.reset_state()
        current_u = initial_input.astype(np.float32).copy()

        for t in range(n_steps):
            self._single_step(current_u)

            flat_state = np.concatenate(self.x_blocks)
            if self.detail_features and self.S > 0:
                delta_list = [
                    self.x_blocks[s - 1] - self.x_blocks[s] for s in range(1, self.S + 1)
                ]
                feat_vec = np.concatenate([flat_state, *delta_list, [1.0]])
            else:
                feat_vec = np.concatenate([flat_state, [1.0]])

            y_t = (self.W_out @ feat_vec).astype(np.float32)
            preds[t] = y_t
            current_u = y_t[:d_in]

        return preds

# Unweighted Erdős–Rényi random graph
from scipy import sparse

n0      = 128               # number of nodes
p_edge  = 4 / n0            # expected degree ≈ 4

rng  = np.random.default_rng(123)
rows = rng.choice(n0, size=int(p_edge * n0 * (n0 - 1) // 2))
cols = rng.choice(n0, size=rows.size)
mask = rows != cols         # avoid self-loops
rows, cols = rows[mask], cols[mask]


# build upper triangle, then symmetrise
adj = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n0, n0))
adj = adj + adj.T
adj[adj > 0] = 1.0          # make unweighted (0/1)



'''
# 4-nearest-neighbour ring lattice (simple and deterministic)
from scipy import sparse
n0  = 128
data, row, col = [], [], []
for i in range(n0):
    for k in (-2, -1, 1, 2):           # 4 neighbours
        j = (i + k) % n0
        row.append(i); col.append(j); data.append(1.0)

adj = sparse.csr_matrix((data, (row, col)), shape=(n0, n0))
'''

'''
# Using NetworkX for more elaborate topologies
import networkx as nx
from scipy import sparse

n0 = 128
G  = nx.random_geometric_graph(n0, radius=0.20, seed=123)
adj = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
adj[adj > 0] = 1.0          # optional: binarise edge weights
'''