import numpy as np    

def compute_valid_prediction_time(y_true, y_pred, t_vals, threshold=0.0001, lambda_max=0.9):
    """
    Compute the Valid Prediction Time (VPT) and compare it to Lyapunov time T_lambda = 1 / lambda_max.
    
    Parameters
    ----------
    y_true : ndarray of shape (N, dim)
        True trajectory over time.
    y_pred : ndarray of shape (N, dim)
        Model's predicted trajectory over time (closed-loop).
    t_vals : ndarray of shape (N,)
        Time values corresponding to the trajectory steps.
    threshold : float, optional
        The error threshold, default is 0.4 as in your snippet.
    lambda_max : float, optional
        Largest Lyapunov exponent. Default=0.9 for Lorenz.
        
    Returns
    -------
    T_VPT : float
        Valid prediction time. The earliest time at which normalized error surpasses threshold
        (or the last time if never surpassed).
    T_lambda : float
        Lyapunov time = 1 / lambda_max
    ratio : float
        How many Lyapunov times the model prediction remains valid, i.e. T_VPT / T_lambda.
    """
    # 1) Average of y_true
    y_mean = np.mean(y_true, axis=0)  # shape (dim,)
    
    # 2) Time-averaged norm^2 of (y_true - y_mean)
    y_centered = y_true - y_mean
    denom = np.mean(np.sum(y_centered**2, axis=1))  # scalar
    
    # 3) Compute the normalized error delta_gamma(t) = ||y_true - y_pred||^2 / denom
    diff = y_true - y_pred
    err_sq = np.sum(diff**2, axis=1)  # shape (N,)
    delta_gamma = err_sq / denom      # shape (N,)
    
    # 4) Find the first time index where delta_gamma(t) exceeds threshold
    idx_exceed = np.where(delta_gamma > threshold)[0]
    if len(idx_exceed) == 0:
        # never exceeds threshold => set T_VPT to the final time
        T_VPT = t_vals[-1]
    else:
        T_VPT = t_vals[idx_exceed[0]]
    
    # 5) Compute T_lambda and ratio
    T_lambda = 1.0 / lambda_max
    ratio = T_VPT / T_lambda

    # 6) Valid Prediction Time (VPT)
    # lambda_max_lorenz = 0.9
    # threshold = 0.01
    print(f"\n--- Valid Prediction Time (VPT) with threshold={threshold}, lambda_max={lambda_max} ---")
    
    return T_VPT, T_lambda, ratio

def mse_dimwise(pred, truth):
    length = min(len(pred), len(truth))
    return np.mean((pred[:length] - truth[:length])**2, axis=0)

def nrmse_dimwise(pred, truth):
    length = min(len(pred), len(truth))
    pred = pred[:length]
    truth = truth[:length]
    mse = np.mean((pred - truth) ** 2, axis=0)
    std = np.std(truth, axis=0)
    std[std == 0] = 1e-8
    return np.sqrt(mse) / std