# --- Keep this at the top, before importing numpy/scipy. ---
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from datetime import datetime
from typing import Iterable, Dict, List, Tuple, Union
import numpy as np
from typing import Optional, Tuple, Dict

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from typing import Tuple, Optional
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
from matplotlib import font_manager
from datetime import datetime
from scipy import stats

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import matplotlib.pyplot as plt

# Configure matplotlib before any plotting code runs.
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Limit per-process BLAS/OpenMP threads to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from datetime import datetime


def tic():
    return time.perf_counter()


def toc(t0, msg=""):
    dt = time.perf_counter() - t0
    if msg:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} elapsed: {dt:.2f}s ({dt / 60:.2f} min)")
    return dt


import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
# import torch
from typing import Tuple, List, Optional, Dict, Union, Callable
from patsy import dmatrix
from itertools import product
from scipy import stats


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


##--- Model setup ---
def Z_design_star(S: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Combine action, intercept, and covariates for data generation.

    Z(S,a) = (a, 1, S1, S2)
    S: (n, 2), a: (n,)
    return: (n, 4)
    """
    if not isinstance(S, torch.Tensor):
        S = torch.as_tensor(S, dtype=torch.float32)
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    S1 = S[:, 0]
    S2 = S[:, 1]
    ones = torch.ones(S1.shape[0], dtype=torch.float32, device=S.device)
    Z = torch.stack([a, ones, S1, S2], dim=1)
    return Z.cpu().numpy()


def Z_design_exp(S: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Combine action, intercept, and covariates for parameter estimation.
    Z(S,a) = (a, 1, S1, S2)
    S: (n, 2), a: (n,)
    return: (n, 4)
    """
    if not isinstance(S, torch.Tensor):
        S = torch.as_tensor(S, dtype=torch.float32)
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    S1 = S[:, 0]
    S2 = S[:, 1]
    ones = torch.ones(S1.shape[0], dtype=torch.float32, device=S.device)
    Z = torch.stack([a, ones, S1, S2], dim=1)
    return Z.cpu().numpy()


def f_star(S, a=2.0, b=1.5):
    """
    f_*(S): nonlinear covariate component, excluding the intercept.

    Parameters:
        S : ndarray, shape (n, d-1)
            State matrix; S[:, 0] is usually the constant column.
        a, b : float
            Coefficients for sine and cosine nonlinearities.

    Returns:
        f_val : ndarray, shape (n,)
            f_*(S_i) for each sample.
    """
    # Start from j=2 in the mathematical notation (Python index 1).
    # Sj = S[:, 1:]  # shape: (n, d-1)

    # Compute the nonlinear sum for each sample.
    if not isinstance(S, torch.Tensor):
        S = torch.as_tensor(S, dtype=torch.float32)
    f_val = a * torch.sin(S + 0.5) + b * torch.cos(S ** 2 - 0.5)
    f_val = torch.sum(f_val, dim=1)
    return f_val.cpu().numpy()


##--- Covariate distribution setup ---


def sample_state(
    ndays: int,
    random_state: Optional[int] = 2025,
    mu: Tuple[float, float] = (0.0, 0.0),
    Sigma: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample covariates S ~ N(mu, Sigma), returned with shape (ndays, 2).
    """
    if Sigma is None:
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=float)
    else:
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.shape != (2, 2):
            raise ValueError("Sigma must be a (2, 2) matrix")
        Sigma = 0.5 * (Sigma + Sigma.T)
    torch.manual_seed(int(random_state) if random_state is not None else 0)
    mu_t = torch.tensor(mu, dtype=torch.float32)
    Sigma_t = torch.tensor(Sigma, dtype=torch.float32)
    L = torch.linalg.cholesky(Sigma_t)
    Z = torch.randn((ndays, 2), dtype=torch.float32)
    S_all = mu_t + Z @ L.T
    return S_all.cpu().numpy()



def sample_state_batch(B: int,
                       ndays: int,
                       random_state: int = 2025,
                       mu: Tuple[float, float] = (0, 0),
                       Sigma: np.ndarray = np.array([[1.0, 0.3],
                                                     [0.3, 1.0]])) -> np.ndarray:
    """
    Fully vectorized sampler that generates (B, ndays, d) in one call.
    All batches share one random stream.
    """
    torch.manual_seed(int(random_state))
    mu = np.asarray(mu)
    d = mu.shape[0]
    L = torch.linalg.cholesky(torch.as_tensor(Sigma, dtype=torch.float32))
    Z = torch.randn((B, ndays, d), dtype=torch.float32)
    mu_t = torch.as_tensor(mu, dtype=torch.float32).view(1, 1, d)
    S_all = mu_t + torch.matmul(Z, L.T)
    return S_all.cpu().numpy()


##--- Data generation ---
def sample_dgp(
        a_all: np.ndarray,
        s_all: np.ndarray,
        theta: np.ndarray = None,
        sigma_pos: float = 1.0,
        sigma_neg: float = 1.0,
        return_per_action: bool = True,
        use_qr: bool = True
):
    n = len(a_all)

    # Design matrix.
    X_all = Z_design_star(s_all, a_all)

    # Parameters.
    if theta is None:
        theta = np.array([1, 1.2, -1.4, 0.8], dtype=float)
    else:
        theta = np.asarray(theta, dtype=float)
        assert theta.shape == (4,)

    # Generate f_* values.
    f_val = f_star(s_all)

    # Heteroskedastic noise.
    sigma_vec = np.where(a_all > 0, float(sigma_pos), float(sigma_neg))
    eps = np.random.standard_normal(n) * sigma_vec

    # Assemble responses.
    R_all = X_all @ theta + f_val + eps

    out = dict(X=X_all, S=s_all, a=a_all, R=R_all, theta=theta,
               f_val=f_val, eps=eps)

    if return_per_action:
        out.update(dict(
            idx_pos=np.where(a_all == 1)[0],
            idx_neg=np.where(a_all == -1)[0],
        ))
    return out


##--- Basis matrix for approximating the nuisance component ---
def make_psi_legendre_tensor(
        S,
        degree=3,
        scaler=None,
        include_intercept=True,
        max_total_degree=None,
        interaction_order=1,
        per_dim_degree=None
):
    """
    Build a multidimensional Legendre tensor-product basis with optional
    intercept, degree limits, and interaction limits. Each column of S is
    scaled to [-1, 1].

    Parameters
    ----
    S : (n, d) ndarray
        Covariate matrix.
    degree : int
        Maximum degree per dimension when per_dim_degree is not provided.
    scaler : dict or None
        {'min': (d,), 'max': (d,)}. If None, fit on S and return the scaler.
    include_intercept : bool
        Whether to include the all-ones intercept column.
    max_total_degree : int or None
        If provided, keep only terms with sum(m_j) <= this value.
    interaction_order : int or None
        If provided, keep only terms with nonzeros(m_j) <= this value.
    per_dim_degree : array-like or None
        Optional length-d vector of maximum degrees for each dimension.

    Returns
    ----
    Psi : (n, p) ndarray
        Tensor-product basis matrix; p depends on the active constraints.
    names : list[str]
        Column names; P0 factors are omitted, and Intercept is optional.
    scaler : dict
        {'min': ..., 'max': ...}, reusable for new data.
    """
    S = torch.as_tensor(S, dtype=torch.float32)
    n, d = S.shape
    if per_dim_degree is None:
        per_dim_degree = np.full(d, int(degree), dtype=int)
    else:
        per_dim_degree = np.asarray(per_dim_degree, dtype=int)
        assert per_dim_degree.shape == (d,), "per_dim_degree length must equal d"
        assert np.all(per_dim_degree >= 0), "Each per-dimension degree must be nonnegative"

    # 1) Scale to [-1, 1].
    if scaler is None:
        col_min = torch.min(S, dim=0).values
        col_max = torch.max(S, dim=0).values
        scaler = {'min': col_min, 'max': col_max}
    else:
        col_min = torch.as_tensor(scaler['min'], dtype=torch.float32)
        col_max = torch.as_tensor(scaler['max'], dtype=torch.float32)
        assert col_min.shape == (d,) and col_max.shape == (d,)
    rng = col_max - col_min
    rng_safe = torch.where(rng == 0.0, torch.tensor(1.0, dtype=torch.float32), rng)
    X = 2.0 * (S - col_min) / rng_safe - 1.0
    mask_const = (rng == 0.0)
    if bool(mask_const.any()):
        X[:, mask_const] = 0.0

    # 2) Precompute P0..P_K for each dimension by recurrence.
    # leg_vals[j][k] is the (n,) vector P_k(s_j) for dimension j.
    leg_vals = []
    for j in range(d):
        K = int(per_dim_degree[j])
        P_prev = torch.ones((n,), dtype=torch.float32)
        if K == 0:
            leg_vals.append([P_prev])
            continue
        xj = X[:, j]
        P_curr = xj.clone()
        cache = [P_prev, P_curr]
        for k in range(1, K):
            P_next = ((2 * k + 1) * xj * P_curr - k * P_prev) / (k + 1)
            cache.append(P_next)
            P_prev, P_curr = P_curr, P_next
        leg_vals.append(cache)

    # 3) Build tensor-product multi-indices and filter by constraints.
    # Each dimension ranges over 0..K_j.
    index_ranges = [range(per_dim_degree[j] + 1) for j in range(d)]
    all_multi_idx = product(*index_ranges)

    def keep_idx(m):
        m = np.asarray(m)
        if not include_intercept and np.all(m == 0):
            return False
        if max_total_degree is not None and m.sum() > max_total_degree:
            return False
        if interaction_order is not None and np.count_nonzero(m) > interaction_order:
            return False
        return True

    filtered_idx = [m for m in all_multi_idx if keep_idx(m)]

    # 4) Generate columns by multiplying dimension-wise factors for each multi-index.
    cols = []
    names = []
    if include_intercept:
        cols.append(torch.ones((n, 1), dtype=torch.float32))
        names.append("Intercept")

    for m in filtered_idx:
        if include_intercept and all(v == 0 for v in m):
            # Intercept has already been added.
            continue
        col = torch.ones(n, dtype=torch.float32)
        parts = []
        for j, kj in enumerate(m):
            # kj may be 0 for P0; otherwise select P_k.
            col = col * leg_vals[j][kj]
            if kj > 0:  # Omit P0 from the displayed column name.
                parts.append(f"P{kj}(s{j + 1})")
        name = "*".join(parts) if parts else "Intercept"
        cols.append(col.reshape(-1, 1))
        names.append(name)

    Psi = torch.hstack(cols) if cols else torch.empty((n, 0))
    return Psi.cpu().numpy(), names, {'min': scaler['min'], 'max': scaler['max']}



def make_psi_legendre_tensor_batch(
        S,
        degree=3,
        scaler=None,
        include_intercept=True,
        max_total_degree=None,
        interaction_order=1,
        per_dim_degree=None
):
    """
    Batched version. S can be (B, n, d) or (n, d), and Psi is returned
    as (B, n, p).
    - Fully vectorized over the large B and n dimensions.
    - Uses small Python loops only over d and the number of basis columns.
    """
    S = torch.as_tensor(S, dtype=torch.float32)
    if S.ndim == 2:
        S = S.unsqueeze(0)
    assert S.ndim == 3

    B, n, d = S.shape
    if per_dim_degree is None:
        per_dim_degree = np.full(d, int(degree), dtype=int)
    else:
        per_dim_degree = np.asarray(per_dim_degree, dtype=int)
        assert per_dim_degree.shape == (d,)
        assert np.all(per_dim_degree >= 0)

    # 1) Scale to [-1, 1] using a global scaler with shape (d,).
    if scaler is None:
        col_min = torch.min(S, dim=(0, 1)).values
        col_max = torch.max(S, dim=(0, 1)).values
        scaler = {'min': col_min, 'max': col_max}
    else:
        col_min = torch.as_tensor(scaler['min'], dtype=torch.float32)
        col_max = torch.as_tensor(scaler['max'], dtype=torch.float32)
        assert col_min.shape == (d,) and col_max.shape == (d,)

    rng = col_max - col_min
    rng_safe = torch.where(rng == 0.0, torch.tensor(1.0, dtype=torch.float32), rng)
    X = 2.0 * (S - col_min.view(1, 1, d)) / rng_safe.view(1, 1, d) - 1.0
    mask_const = (rng == 0.0)
    if bool(mask_const.any()):
        X[:, :, mask_const] = 0.0

    # 2) Precompute P0..P_K per dimension, vectorized to (B, n).
    # leg_vals[j][k] has shape (B, n).
    leg_vals = []
    for j in range(d):
        K = int(per_dim_degree[j])
        P0 = torch.ones((B, n), dtype=torch.float32)
        if K == 0:
            leg_vals.append([P0])
            continue
        xj = X[:, :, j]
        P1 = xj.clone()
        cache = [P0, P1]
        Pkm1, Pk = P0, P1
        for k in range(1, K):
            Pkp1 = ((2 * k + 1) * xj * Pk - k * Pkm1) / (k + 1)
            cache.append(Pkp1)
            Pkm1, Pk = Pk, Pkp1
        leg_vals.append(cache)

    # 3) Generate multi-indices and filter by constraints.
    index_ranges = [range(per_dim_degree[j] + 1) for j in range(d)]
    all_multi_idx = product(*index_ranges)

    def keep_idx(m):
        m = np.asarray(m)
        if not include_intercept and np.all(m == 0):
            return False
        if max_total_degree is not None and m.sum() > max_total_degree:
            return False
        if interaction_order is not None and np.count_nonzero(m) > interaction_order:
            return False
        return True

    filtered_idx = [m for m in all_multi_idx if keep_idx(m)]

    # 4) Generate columns by dimension-wise multiplication for each multi-index.
    cols = []
    names = []
    if include_intercept:
        cols.append(torch.ones((B, n, 1), dtype=torch.float32))
        names.append("Intercept")

    for m in filtered_idx:
        if include_intercept and all(v == 0 for v in m):
            continue
        col = torch.ones((B, n), dtype=torch.float32)
        parts = []
        for j, kj in enumerate(m):
            col = col * leg_vals[j][kj]
            if kj > 0:
                parts.append(f"P{kj}(s{j + 1})")
        name = "*".join(parts) if parts else "Intercept"
        cols.append(col[..., None])
        names.append(name)

    Psi = torch.concatenate(cols, dim=2) if cols else torch.empty((B, n, 0))
    return Psi.cpu().numpy(), names, {'min': scaler['min'].cpu().numpy(), 'max': scaler['max'].cpu().numpy()}


##-- Build the action space from length-n action sequences. ---
def make_structured_action_space(n=14, random_state=None):
    """
    Generate a structured action-space matrix with shape (n+1, n):
      - Row 0: all -1
      - Row k: randomly choose k positions as +1 and keep the rest at -1
      - Row n: all +1

    Parameters
    ----
    n : int
        Number of days, i.e. number of columns.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    ----
    actions_space : ndarray, shape (n+1, n)
        Structured action-space matrix.
    """
    rng = np.random.default_rng(random_state)
    actions_space = np.full((n + 1, n), -1, dtype=int)

    for k in range(1, n + 1):
        idx = rng.choice(n, size=k, replace=False)
        actions_space[k, idx] = 1

    return actions_space


def make_structured_action_space_batch(
        B: int,
        n: int = 14,
        random_state: int = None
) -> np.ndarray:
    """
    Fast vectorized version with a shared random stream and distinct batches.
    Returns (B, n+1, n).
    """
    rng = np.random.default_rng(random_state)
    actions_all = np.full((B, n + 1, n), -1, dtype=int)

    # Generate random permutation tensors (B, n, n), shuffling column indices by row.
    perms = np.array([rng.permutation(n) for _ in range(B * n)]).reshape(B, n, n)
    # Fill the +1 regions.
    for k in range(1, n + 1):
        # For each batch b, set the first k sampled positions in row k to +1.
        idx = perms[:, k - 1, :k]  # shape (B,k)
        for b in range(B):
            actions_all[b, k, idx[b]] = 1
    return actions_all



def Refine_Q_robust_obj_fun_v1_batch_flat(
        L_basis, ndays,
        delta_a,      # (B*K,) or (B,K) or (N,1); process in chunks for large B.
        Delta_a,      # (B*K,d) or (B,K,d)
        Gamma_a,      # (B*K,L) or (B,K,L)
        Sigma_mat,    # (d,d)
        Xi_mat,       # (L, d+1)
        Utilde_mat,   # (L, L-d-1)
        nu_factor,
        eps=1e-2,
        device=None,
        return_numpy=True,
):
    """
    GPU-compatible robust objective function implemented with torch.

    Returns:
        obj_term, var_term, bias_term
        - If return_numpy=True: numpy.ndarray objects with shape (N,)
        - Otherwise: torch.Tensor objects with shape (N,)
    """
    # -------- Select device --------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------- Helper: convert to tensor --------
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        else:
            return torch.as_tensor(x, device=device, dtype=torch.float32)

    # Convert inputs to tensors.
    delta_t  = to_tensor(delta_a)
    Delta_t  = to_tensor(Delta_a)
    Gamma_t  = to_tensor(Gamma_a)
    Sigma_t  = to_tensor(Sigma_mat)
    Xi_t     = to_tensor(Xi_mat)
    Utilde_t = to_tensor(Utilde_mat)

    # -------- Normalize inputs to a flat N dimension --------
    # delta: (N,)
    if delta_t.ndim == 2:
        delta_t = delta_t.t().reshape(-1)
    else:
        delta_t = delta_t.reshape(-1)

    # Delta: (N,d)
    if Delta_t.ndim == 3:
        N, d = Delta_t.shape[0] * Delta_t.shape[1], Delta_t.shape[2]
        Delta_t = Delta_t.permute(1, 0, 2).reshape(-1, d)
    else:
        d = Delta_t.shape[1]
        Delta_t = Delta_t.reshape(-1, d)

    # Gamma: (N,L)
    if Gamma_t.ndim == 3:
        N, L = Gamma_t.shape[0] * Gamma_t.shape[1], Gamma_t.shape[2]
        Gamma_t = Gamma_t.permute(1, 0, 2).reshape(-1, L)
    else:
        L = Gamma_t.shape[1]
        Gamma_t = Gamma_t.reshape(-1, L)

    # -------- Shape checks, identical on GPU and CPU --------
    assert Sigma_t.shape == (d, d)
    assert Xi_t.shape == (L, d + 1)
    assert Utilde_t.shape == (L, L - d - 1)

    # -------- Precompute --------
    Sigma_inv = torch.linalg.inv(Sigma_t)   # (d,d)

    # Δᵀ Σ^{-1} Δ -> (N,)
    Delta_norm = torch.einsum('nd,dd,nd->n', Delta_t, Sigma_inv, Delta_t)

    # v = [delta; Delta] -> (N,1+d)
    v = torch.cat([delta_t[:, None], Delta_t], dim=1)   # (N, 1+d)

    # Σ^{-1}_aug: (1+d,1+d)
    eye_1 = torch.eye(1, device=device, dtype=torch.float32)
    Sigma_inv_aug = torch.block_diag(eye_1, Sigma_inv)  # (1+d,1+d)

    # w = v Σ^{-1}_augᵀ -> (N,1+d)
    w = v @ Sigma_inv_aug.T

    # t = w Xiᵀ -> (N,L)
    t = w @ Xi_t.T

    # Γ̃ = (Γ - t) U -> (N, L-d-1)
    Gamma_tilde = (Gamma_t - t) @ Utilde_t

    # ||Γ̃||² -> (N,)
    Gamma_tilde_norm = (Gamma_tilde * Gamma_tilde).sum(dim=1)

    # -------- Variance term and bias upper bound --------
    # denom = ndays - (1/ndays) * (delta^2 + ||Delta||_{Sigma^{-1}}^2)
    ndays_t = torch.as_tensor(ndays, device=device, dtype=torch.float32)
    nu_t    = torch.as_tensor(nu_factor, device=device, dtype=torch.float32)
    Lb_t    = torch.as_tensor(L_basis, device=device, dtype=torch.float32)

    denom = ndays_t - (1.0 / ndays_t) * (delta_t ** 2 + Delta_norm)

    # Numerical guard against nonpositive or tiny denominators.
    denom = torch.clamp(denom, min=eps)

    var_term  = nu_t / denom                # (N,)
    bias_term = Lb_t * Gamma_tilde_norm / (denom ** 2)  # (N,)
    obj_term  = var_term + bias_term

    if return_numpy:
        obj_np  = obj_term.detach().cpu().numpy()
        var_np  = var_term.detach().cpu().numpy()
        bias_np = bias_term.detach().cpu().numpy()
        return obj_np, var_np, bias_np
    else:
        return obj_term, var_term, bias_term


def Refine_Q_nonrobust_obj_fun_v1_batch_flat(
        L_basis, ndays,
        delta_a,    # (B*K,) or (B,K)
        Delta_a,    # (B*K,d) or (B,K,d)
        Gamma_a,    # unused
        Sigma_mat,  # (d,d)
        Xi_mat,     # unused
        Utilde_mat, # unused
        nu_factor,
        eps=1e-8,
        device=None,
        return_numpy=True,
):
    """
    GPU-compatible non-DP-sequence objective:
    unbalanced_term = delta^2 + Delta^T Sigma^{-1} Delta.

    - The call signature is kept compatible with the robust objective.
    - If a GPU is available, the computation runs on CUDA by default.
    """
    # -------- Select device --------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        else:
            return torch.as_tensor(x, device=device, dtype=torch.float32)

    delta_t = to_tensor(delta_a)
    Delta_t = to_tensor(Delta_a)
    Sigma_t = to_tensor(Sigma_mat)

    # -------- Normalize inputs to a flat N dimension --------
    # delta: (N,)
    if delta_t.ndim == 2:
        delta_t = delta_t.t().reshape(-1)
    else:
        delta_t = delta_t.reshape(-1)

    # Delta: (N,d)
    if Delta_t.ndim == 3:
        N, d = Delta_t.shape[0] * Delta_t.shape[1], Delta_t.shape[2]
        Delta_t = Delta_t.permute(1, 0, 2).reshape(-1, d)
    else:
        d = Delta_t.shape[1]
        Delta_t = Delta_t.reshape(-1, d)

    assert Sigma_t.shape == (d, d)

    # -------- Delta^T Sigma^{-1} Delta --------
    Sigma_inv = torch.linalg.inv(Sigma_t)                      # (d,d)
    Delta_norm = torch.einsum('nd,dd,nd->n', Delta_t, Sigma_inv, Delta_t)

    # -------- Imbalance term --------
    unbalanced = delta_t**2 + Delta_norm
    unbalanced = torch.clamp(unbalanced, min=eps)

    if return_numpy:
        return unbalanced.detach().cpu().numpy()
    else:
        return unbalanced







def est_u_tilde(C_hat):
    C = torch.as_tensor(C_hat, dtype=torch.float32)
    U_full, s, Vt = torch.linalg.svd(C, full_matrices=True)
    tol = float(torch.max(s)) * max(C.shape) * np.finfo(np.float32).eps if s.numel() > 0 else 0.0
    r = int(torch.sum(s > tol).item())
    U_tilde = U_full[:, r:]
    return U_tilde.cpu().numpy(), r



def ols_fit(X, y, robust=True):
    """
    OLS for y ~ X, where X includes your columns (e.g., a, 1, S1, S2).
    Returns: beta_hat, se_homosked, se_robust(optional), stats dict
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, k = X.shape

    # OLS coefficients (use lstsq for stability)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Residuals and sums of squares
    y_hat = X @ beta_hat
    e = y - y_hat
    sse = float(e @ e)
    sst = float(((y - y.mean()) @ (y - y.mean())))  # with intercept, this is standard SST
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    df = max(n - k, 1)
    sigma2 = sse / df

    # (X'X)^(-1)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)  # pinv is safer if collinearity

    # Homoskedastic SE
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))

    # HC0 DP_seq SE (White)
    se_robust = None
    if robust:
        # meat = X' diag(e^2) X = sum_i e_i^2 x_i x_i'
        meat = (X * e[:, None]).T @ (X * e[:, None])
        var_beta_hc0 = XtX_inv @ meat @ XtX_inv
        se_robust = np.sqrt(np.diag(var_beta_hc0))

    stats = {
        "n": n, "k": k, "df_resid": df,
        "SSE": sse, "SST": sst, "sigma2": sigma2,
        "R2": r2, "Adj_R2": 1 - (1 - r2) * (n - 1) / df if sst > 0 else np.nan
    }
    return beta_hat, se, se_robust, e, stats


def prespecified_params_fun(M_rept,
                            ndays=500_000,
                            degree=3,
                            include_intercept=True,
                            interaction_order=1,
                            print_every=10):
    """
    Lightweight estimator returning
    (ATE_mc_true, C_hat_mean, Xi_mat, Sigma_mat, Utilde_mat, fixed_scaler, L_basis).
    - Fit the scaler once so Psi is consistent.
    - Use online averages for numerical stability and memory efficiency.
    - Print concise progress updates.
    """
    # 1) Fit the scaler once.
    S_scaler = sample_state(ndays=ndays)
    _, _, fixed_scaler = make_psi_legendre_tensor(
        S_scaler, degree=degree, include_intercept=include_intercept, interaction_order=interaction_order
    )

    # 2) Constants.
    a_all_treated = torch.ones(ndays, dtype=torch.float32)
    a_all_control = -torch.ones(ndays, dtype=torch.float32)
    Sigma_mat = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float32)

    # 3) Online-average containers.
    ATE_mean = 0.0
    C_hat_mean = None

    ts = lambda: datetime.now().strftime('%H:%M:%S')

    for t in range(M_rept):
        if (t + 1) % print_every == 0:
            print(f"[{ts()}] MC {t + 1}/{M_rept} ...")

        # Sample states.
        S_all = sample_state(ndays=ndays)

        # Mean reward difference under the two actions.
        R_treated = sample_dgp(a_all=a_all_treated.cpu().numpy(), s_all=S_all)["R"]
        R_control = sample_dgp(a_all=a_all_control.cpu().numpy(), s_all=S_all)["R"]
        ATE_t = float(np.mean(R_treated) - np.mean(R_control))
        ATE_mean += (ATE_t - ATE_mean) / (t + 1)

        # Generate Psi(S) using the fixed scaler.
        Psi_S, _, _ = make_psi_legendre_tensor(
            S_all, degree=degree, scaler=fixed_scaler,
            include_intercept=include_intercept, interaction_order=interaction_order
        )

        # Design matrix; keep state-related columns only.
        X_all = Z_design_exp(S_all, a_all_treated.cpu().numpy())
        S_aug = X_all[:, 1:]

        # Monte Carlo approximation of C_hat = E[Psi(S) S^T].
        C_hat = (Psi_S.T @ S_aug) / float(ndays)

        if C_hat_mean is None:
            C_hat_mean = C_hat.astype(float)
        else:
            C_hat_mean += (C_hat - C_hat_mean) / (t + 1)

    # Derived quantities.
    Xi_mat = C_hat_mean
    Utilde_mat, _ = est_u_tilde(C_hat_mean)
    L_basis = Utilde_mat.shape[0]

    # Concise summary output.
    np.set_printoptions(precision=6, suppress=True)
    print("\n=== Summary ===")
    print(f"ATE_mc_true: {ATE_mean:.6f}")
    print(f"C_hat_mean (shape {C_hat_mean.shape}):\n{C_hat_mean}")
    print(f"Xi_mat (shape {Xi_mat.shape}) = C_hat_mean")
    print(f"Sigma_mat:\n{Sigma_mat}")
    print(f"Utilde_mat (shape {Utilde_mat.shape}):\n{Utilde_mat}")
    print("===============\n")

    return ATE_mean, C_hat_mean, Xi_mat, Sigma_mat, Utilde_mat, fixed_scaler, L_basis



# ---------------- Scalers ----------------
class MinMaxScalerX:
    def __init__(self, min_val, max_val, eps=1e-12):
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    @classmethod
    def fit(cls, X_tr):
        min_val = X_tr.min(axis=0, keepdims=True)
        max_val = X_tr.max(axis=0, keepdims=True)
        return cls(min_val, max_val)

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val + self.eps)

    def inverse_transform(self, X_scaled):
        return X_scaled * (self.max_val - self.min_val) + self.min_val


# Target scaler that preserves relative scale for values near zero.
class RobustTargetScalerY:
    def __init__(self, min_val, max_val, eps=1e-12):
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.eps = eps

    @classmethod
    def fit(cls, y_tr):
        min_val = float(np.min(y_tr))
        max_val = float(np.max(y_tr))
        return cls(min_val, max_val)

    def transform(self, y):
        # Use min-max scaling to preserve relative relationships in the raw target.
        return (y - self.min_val) / (self.max_val - self.min_val + self.eps)

    def inverse_transform(self, y_scaled):
        return y_scaled * (self.max_val - self.min_val) + self.min_val


def create_model(in_dim, device='cpu'):
    """
    Create the same neural-network architecture used during training.

    Parameters:
    - in_dim: input dimension
    - device: device string, e.g. 'cpu' or 'cuda'

    Returns:
    - model: PyTorch model
    """
    hidden1, hidden2, hidden3 = 128,64,32  # Hidden-layer widths used in training.

    model = nn.Sequential(
        # Layer 1: input -> hidden1.
        nn.Linear(in_dim, hidden1),
        nn.BatchNorm1d(hidden1),
        nn.ReLU(),
        nn.Dropout(0.1),  # Conservative dropout rate.

        # Layer 2: hidden1 -> hidden2.
        nn.Linear(hidden1, hidden2),
        nn.BatchNorm1d(hidden2),
        nn.ReLU(),
        nn.Dropout(0.1),

        # Layer 3: hidden2 -> hidden3.
        nn.Linear(hidden2, hidden3),
        nn.BatchNorm1d(hidden3),
        nn.ReLU(),
        nn.Dropout(0.1),

        # Layer 4: hidden3 -> output.
        nn.Linear(hidden3, 1)
    ).to(device)

    return model



@torch.no_grad()
def eval_metrics_std_space(model, X_t, y_t, device='cpu'):
    """
    Evaluate model performance in standardized space.
    """
    model.eval()
    preds_std, ys_std = [], []
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=False)
    for xb, yb in loader:
        xb = xb.to(device);
        yb = yb.to(device)
        pred_std = model(xb)
        preds_std.append(pred_std.cpu());
        ys_std.append(yb.cpu())
    p_std = torch.cat(preds_std);
    y_std = torch.cat(ys_std);
    mse_std = torch.mean((p_std - y_std) ** 2).item()
    mae_std = torch.mean(torch.abs(p_std - y_std)).item()
    var_y = torch.var(y_std, unbiased=False).item()
    r2 = 1.0 - (mse_std / (var_y + 1e-12))
    nrmse = float(np.sqrt(mse_std / (var_y + 1e-12)))
    return dict(mse_std=mse_std, mae_std=mae_std, r2=r2, nrmse=nrmse)


@torch.no_grad()
def eval_metrics_log_space(model, X_t, y_t, device='cpu'):
    """
    Evaluate model performance in log(y_orig) space.
    This function assumes that both y_t and model predictions are in
    log(y_orig) space. MSE, MAE, R2, and NRMSE are computed in that space.
    """
    model.eval()
    preds_log, ys_log = [], []
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=False)
    for xb, yb_log in loader: # Renamed yb to yb_log for clarity
        xb = xb.to(device)
        yb_log = yb_log.to(device) # yb_log is the true log(y_orig) value
        pred_log = model(xb)       # pred_log is the predicted log(y_orig) value
        preds_log.append(pred_log.cpu())
        ys_log.append(yb_log.cpu())
    p_log = torch.cat(preds_log)
    y_log = torch.cat(ys_log) # y_log is the true log(y_orig) tensor

    # Calculate metrics directly in the log(y_orig) space
    mse_log = torch.mean((p_log - y_log) ** 2).item()
    mae_log = torch.mean(torch.abs(p_log - y_log)).item()
    var_y_log = torch.var(y_log, unbiased=False).item() # Variance of true log(y_orig)
    r2_log = 1.0 - (mse_log / (var_y_log + 1e-12))
    nrmse_log = float(np.sqrt(mse_log / (var_y_log + 1e-12)))

    return dict(mse_log=mse_log, mae_log=mae_log, r2_log=r2_log, nrmse_log=nrmse_log)

def baseline_metrics(y_train, y_test):
    mu = float(np.mean(y_train))
    mse = float(np.mean((y_test - mu) ** 2))
    mae = float(np.mean(np.abs(y_test - mu)))
    var = float(np.var(y_test))
    r2 = 1.0 - (mse / (var + 1e-12))
    nrmse = float(np.sqrt(mse / (var + 1e-12)))
    return dict(mse=mse, mae=mae, r2=r2, nrmse=nrmse)


def _to_2d(x):
    x = np.asarray(x)
    if x.ndim == 0: return x.reshape(1, 1)
    if x.ndim == 1: return x.reshape(-1, 1)
    return x.reshape(-1, x.shape[-1])

def _names(base, c):
    return [base] if c == 1 else [f"{base}_{i+1}" for i in range(c)]

def save_features_labels(
    delta_a_flat,      # (n,1) or (n,)
    Delta_a_flat,      # (n,d1)
    Gamma_a_flat,      # (n,d2)
    Out_Q_res,         # (n,1) or (n,c)
    path="dataset.txt" # Default tab-delimited .txt; .txt.gz is compressed automatically.
):
    # Normalize all inputs to 2D arrays.
    delta_a_flat = _to_2d(delta_a_flat)   # (n, ?)
    Delta_a_flat = _to_2d(Delta_a_flat)   # (n, d1)
    Gamma_a_flat = _to_2d(Gamma_a_flat)   # (n, d2)
    Out_Q_res    = _to_2d(Out_Q_res)      # (n, c)

    # Row-count consistency checks.
    n = delta_a_flat.shape[0]
    assert Delta_a_flat.shape[0] == n, "Delta_a_flat row count mismatch"
    assert Gamma_a_flat.shape[0] == n, "Gamma_a_flat row count mismatch"
    assert Out_Q_res.shape[0]    == n, "Out_Q_res row count mismatch; labels and features must align"

    # Column names.
    cols = []
    cols += _names("delta_a_flat", delta_a_flat.shape[1])
    cols += _names("Delta_a_flat", Delta_a_flat.shape[1])
    cols += _names("Gamma_a_flat", Gamma_a_flat.shape[1])
    cols += _names("Out_Q_res",    Out_Q_res.shape[1])

    # Concatenate features and labels.
    data = np.hstack([delta_a_flat, Delta_a_flat, Gamma_a_flat, Out_Q_res])

    # Build a DataFrame while preserving numeric column values.
    df = pd.DataFrame(data, columns=cols)

    # Save as a tab-delimited text file; .gz suffix triggers gzip compression.
    df.to_csv(path, sep="\t", index=False, encoding="utf-8",
              float_format="%.6g", compression="infer")
    print(f"Saved to {path} with shape {df.shape}")


def Q_iter_ger_data(
    checkpoint,
    n_exp,
    B,
    seed,
    scaler,
    act_share_pos,
    act_share_neg,
    S_exp_mat_share_pos,
    S_exp_mat_share_neg,
    Psi_exp_mat_share_pos,
    Psi_exp_mat_share_neg,
    obj_type="var",
    BK_chunk_size=256,        # Conservative default for 10 GB GPUs.
    B_share_chunk_size=512,  # Additional chunking over shared samples.
):
    """
    Iteratively generate the Q-value dataset for day n_exp and use the
    previous-round Q network for prediction.

    Shape conventions:
    - sample_state_batch(B, n_exp, ...) -> (B, n_exp, d_s)
    - make_psi_legendre_tensor_batch(...) -> (B, n_exp, L)
    - make_structured_action_space_batch(B, n_exp, ...) -> (B, K, n_exp)

    Computed summaries:
    - delta_a: (B, K)
    - Delta_a: (B, K, d_s)
    - Gamma_a: (B, K, L)

    Then flatten (B, K, *) to (B*K, *) and combine with shared samples
    of shape (B_share, *).

    Key memory optimizations:
    - Keep BK chunking.
    - Add chunking over B_share.
    - Avoid constructing a large (bk * B_share, in_dim) tensor at once.
    """

    # ------------------------
    # 0. Device and random seed.
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Q_iter_ger_data] device = {device}, n_exp = {n_exp}, B = {B}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ------------------------
    # 1. Construct S_exp, Psi_exp, and act_exp.
    # ------------------------
    S_exp_np = sample_state_batch(B, n_exp, random_state=seed)
    if not (S_exp_np.ndim == 3 and S_exp_np.shape[0] == B and S_exp_np.shape[1] == n_exp):
        raise ValueError(
            f"sample_state_batch returned an unexpected shape: {S_exp_np.shape}; "
            f"expected (B={B}, n_exp={n_exp}, d_s)"
        )

    Psi_exp_np, _, _ = make_psi_legendre_tensor_batch(
        S_exp_np,
        degree=3,
        scaler=scaler,
        include_intercept=True,
        interaction_order=1,
    )

    act_exp_np = make_structured_action_space_batch(B, n_exp, random_state=seed)

    act_exp = torch.as_tensor(act_exp_np, dtype=torch.float32, device=device)
    S_exp = torch.as_tensor(S_exp_np, dtype=torch.float32, device=device)
    Psi_exp = torch.as_tensor(Psi_exp_np, dtype=torch.float32, device=device)

    if act_exp.ndim != 3:
        raise ValueError(f"make_structured_action_space_batch did not return a 3D tensor: {act_exp.shape}")
    B_act, K, T_act = act_exp.shape
    if B_act != B or T_act != n_exp:
        raise ValueError(
            f"act_exp.shape = {act_exp.shape}; expected (B={B}, K=?, n_exp={n_exp})"
        )

    # ------------------------
    # 2. Compute delta_a, Delta_a, and Gamma_a.
    # ------------------------
    delta_a = act_exp.sum(dim=2)
    Delta_a = torch.matmul(act_exp, S_exp)
    Gamma_a = torch.matmul(act_exp, Psi_exp)

    BK = B * K
    delta_a_flat = delta_a.view(BK, 1)
    Delta_a_flat = Delta_a.view(BK, Delta_a.shape[-1])
    Gamma_a_flat = Gamma_a.view(BK, Gamma_a.shape[-1])

    print(
        f"[Q_iter_ger_data] K = {K}, BK = {BK}, "
        f"delta_a_flat = {delta_a_flat.shape}, "
        f"Delta_a_flat = {Delta_a_flat.shape}, "
        f"Gamma_a_flat = {Gamma_a_flat.shape}"
    )

    # ------------------------
    # 3. Move shared samples to the selected device.
    # ------------------------
    act_share_pos_t = torch.as_tensor(act_share_pos, dtype=torch.float32, device=device)
    act_share_neg_t = torch.as_tensor(act_share_neg, dtype=torch.float32, device=device)

    S_share_pos_t = torch.as_tensor(S_exp_mat_share_pos, dtype=torch.float32, device=device)
    S_share_neg_t = torch.as_tensor(S_exp_mat_share_neg, dtype=torch.float32, device=device)

    Psi_share_pos_t = torch.as_tensor(Psi_exp_mat_share_pos, dtype=torch.float32, device=device)
    Psi_share_neg_t = torch.as_tensor(Psi_exp_mat_share_neg, dtype=torch.float32, device=device)

    B_share = act_share_pos_t.shape[0]
    print(f"[Q_iter_ger_data] B_share = {B_share}")

    # ------------------------
    # 4. Restore x_scaler and model from the checkpoint.
    # ------------------------
    x_scaler_min = checkpoint["x_scaler_min"]
    x_scaler_max = checkpoint["x_scaler_max"]

    x_min = torch.as_tensor(x_scaler_min, dtype=torch.float32, device=device)
    x_max = torch.as_tensor(x_scaler_max, dtype=torch.float32, device=device)
    eps = 1e-12

    in_dim = x_min.shape[1]
    print(f"[Q_iter_ger_data] in_dim = {in_dim}")

    Q_net_last_model = create_model(in_dim, device=device)
    Q_net_last_model.load_state_dict(checkpoint["model_state_dict"])
    Q_net_last_model.to(device)
    Q_net_last_model.eval()

    # ------------------------
    # 5. Two-level chunking over BK and B_share.
    # ------------------------
    mean_chunks = []
    n_chunks = (BK + BK_chunk_size - 1) // BK_chunk_size
    processed = 0

    print(
        f"[Q_iter_ger_data] Start forward in {n_chunks} chunks, "
        f"BK_chunk_size = {BK_chunk_size}, "
        f"B_share_chunk_size = {B_share_chunk_size}"
    )

    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, BK, BK_chunk_size)):
            end = min(start + BK_chunk_size, BK)
            bk = end - start

            delta_chunk = delta_a_flat[start:end]
            Delta_chunk = Delta_a_flat[start:end]
            Gamma_chunk = Gamma_a_flat[start:end]

            # Accumulate sum(min) to avoid materializing all B_share pairs at once.
            sum_min = torch.zeros((bk, 1), device=device, dtype=torch.float32)
            counted = 0

            for js in range(0, B_share, B_share_chunk_size):
                je = min(js + B_share_chunk_size, B_share)
                bj = je - js

                act_pos_sub = act_share_pos_t[js:je]         # (bj,1)
                act_neg_sub = act_share_neg_t[js:je]
                S_pos_sub = S_share_pos_t[js:je]             # (bj,d_s)
                S_neg_sub = S_share_neg_t[js:je]
                Psi_pos_sub = Psi_share_pos_t[js:je]         # (bj,L)
                Psi_neg_sub = Psi_share_neg_t[js:je]

                delta_pos = (delta_chunk + act_pos_sub.T).reshape(-1, 1)
                delta_neg = (delta_chunk + act_neg_sub.T).reshape(-1, 1)

                Delta_pos = (Delta_chunk[:, None, :] + S_pos_sub[None, :, :]).reshape(-1, Delta_chunk.shape[1])
                Delta_neg = (Delta_chunk[:, None, :] + S_neg_sub[None, :, :]).reshape(-1, Delta_chunk.shape[1])

                Gamma_pos = (Gamma_chunk[:, None, :] + Psi_pos_sub[None, :, :]).reshape(-1, Gamma_chunk.shape[1])
                Gamma_neg = (Gamma_chunk[:, None, :] + Psi_neg_sub[None, :, :]).reshape(-1, Gamma_chunk.shape[1])

                X_pos = torch.cat([delta_pos, Delta_pos, Gamma_pos], dim=1)
                X_neg = torch.cat([delta_neg, Delta_neg, Gamma_neg], dim=1)

                assert X_pos.shape[1] == in_dim
                assert X_neg.shape[1] == in_dim

                X_pos_s = (X_pos - x_min) / (x_max - x_min + eps)
                X_neg_s = (X_neg - x_min) / (x_max - x_min + eps)

                pred_pos = Q_net_last_model(X_pos_s)
                pred_neg = Q_net_last_model(X_neg_s)

                pred_pos_2d = pred_pos.view(bk, bj)
                pred_neg_2d = pred_neg.view(bk, bj)

                pred_min_2d = torch.minimum(pred_pos_2d, pred_neg_2d)

                sum_min += pred_min_2d.sum(dim=1, keepdim=True)
                counted += bj

            mean_chunk = sum_min / float(counted)
            mean_chunks.append(mean_chunk)

            processed += bk
            if (
                chunk_idx == 0
                or chunk_idx == n_chunks - 1
                or (chunk_idx + 1) % max(1, n_chunks // 10) == 0
            ):
                print(
                    f"[Q_iter_ger_data] chunk {chunk_idx+1}/{n_chunks} done, "
                    f"processed {processed}/{BK}"
                )

    mean_Y_pred_std = torch.cat(mean_chunks, dim=0)
    assert mean_Y_pred_std.shape[0] == BK

    print(f"[Q_iter_ger_data] Prediction finished. mean_Y_pred_std shape: {mean_Y_pred_std.shape}")

    # ------------------------
    # 6. Save dataset.
    # ------------------------
    Out_Q_res = mean_Y_pred_std.cpu().numpy()
    delta_np = delta_a_flat.cpu().numpy()
    Delta_np = Delta_a_flat.cpu().numpy()
    Gamma_np = Gamma_a_flat.cpu().numpy()

    save_features_labels(
        delta_np,
        Delta_np,
        Gamma_np,
        Out_Q_res,
        path=f"Qval_dataset_{n_exp}_{obj_type}.txt",
    )

    print(f"[Q_iter_ger_data] Data saved to Qval_dataset_{n_exp}_{obj_type}.txt")

    return float(Out_Q_res.mean())







def custom_selection(obj_bk_pos, obj_bk_neg):
    """
    Select values row-wise from two (B,) arrays using the project rule.

    Args:
        obj_bk_pos (np.ndarray): Array with shape (B,).
        obj_bk_neg (np.ndarray): Array with shape (B,).

    Returns:
        np.ndarray: Result array with shape (B,). Rows with two negative
        candidates receive the global maximum positive candidate when available.
    """
    # Ensure inputs are NumPy arrays.
    obj_bk_pos = np.asarray(obj_bk_pos)
    obj_bk_neg = np.asarray(obj_bk_neg)

    # Check shapes.
    assert obj_bk_pos.shape == obj_bk_neg.shape, "Input arrays must have the same shape"

    B = obj_bk_pos.shape[0]

    # Initialize with NaN to make unresolved edge cases visible.
    result = np.full(B, np.nan)

    # Case 1: both candidates are positive.
    mask_both_pos = (obj_bk_pos > 0) & (obj_bk_neg > 0)
    # For Case 1 rows, use the smaller positive value.
    result[mask_both_pos] = np.minimum(obj_bk_pos[mask_both_pos], obj_bk_neg[mask_both_pos])

    # Case 2: exactly one candidate is positive.
    mask_one_pos_one_neg = (obj_bk_pos > 0) != (obj_bk_neg > 0) # XOR equivalent.
    # For Case 2 rows, choose the positive candidate.

    # np.where(condition, x, y) returns x where condition is True, y where False
    result[mask_one_pos_one_neg] = np.where(
        obj_bk_pos[mask_one_pos_one_neg] > 0,
        obj_bk_pos[mask_one_pos_one_neg],
        obj_bk_neg[mask_one_pos_one_neg]
    )

    # Case 3: both candidates are negative.
    mask_both_neg = (obj_bk_pos < 0) & (obj_bk_neg < 0)

    # Check whether Case 3 occurs.
    if np.any(mask_both_neg):
        # Gather positive values as fallback candidates for Case 3.
        all_pos_values = np.concatenate([
            obj_bk_pos[obj_bk_pos > 0],
            obj_bk_neg[obj_bk_neg > 0]
        ])

        if all_pos_values.size > 0:
            # Use the global maximum positive value when available.
            global_max_pos = np.max(all_pos_values)
            # Assign it to all Case 3 rows.
            result[mask_both_neg] = global_max_pos
        else:
            # If no positive value exists, keep Case 3 rows as NaN.
            # print("Warning: All values in obj_bk_pos and obj_bk_neg are <= 0. Result contains NaN.")
            pass # Keep result[mask_both_neg] as NaN.

    # Remaining NaN values should only occur for NaN inputs or unresolved edge cases.
    if np.any(np.isnan(result)):
        print("Warning: Some entries in the result are NaN. This might indicate input values were NaN or a logical edge case.")

    return result
