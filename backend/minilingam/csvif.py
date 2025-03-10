#!/usr/bin/env python3

import argparse
import numpy as np
import networkx as nx
import sys
import time
import itertools
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR

def main():
    parser = argparse.ArgumentParser(description='Discover causal relationships in time series data using FastLiNGAM.')
    parser.add_argument('-i', '--input', default=None,
                        help='Input CSV file path (default: standard input)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file path (default: standard output)')
    parser.add_argument('-t', '--time', default=None,
                        help='Output time file path (default: none)')
    parser.add_argument('-w', '--weighted_edgelist', default=None,
                        help='Output weighted edgelist file path (default: none)')
    parser.add_argument('--max_lags', type=int, default=1,
                        help='Maximum number of lags to consider (default: 1)')
    parser.add_argument('-l', '--weight_threshold', type=float, default=0.05,
                        help='Lowest weight to be considered as an edge (default: 0.05)')
    
    # Show help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    analyze_csv_data(args.input, args.output, args.time, args.weighted_edgelist, 
                    args.max_lags, args.weight_threshold)

def analyze_csv_data(input_file, output_file, time_file, weighted_edgelist_file, max_lags, weight_threshold):
    if input_file == 'stdin':
        raw_data = np.genfromtxt(sys.stdin, delimiter=',', skip_header=1)
        headers = next(sys.stdin).strip().split(',')
    else:
        raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
        headers = np.genfromtxt(input_file, delimiter=',', max_rows=1, dtype=str)
    
    data = raw_data
    start_time = time.time()
    summary_matrix = compute_causal_matrix(data, max_lags)
    computation_time = time.time() - start_time
    
    if time_file:
        with open(time_file, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    G = nx.DiGraph()
    for header in headers:
        G.add_node(header)
    
    for i in range(len(headers)):
        for j in range(len(headers)):
            if summary_matrix[i, j] > weight_threshold:
                weight = summary_matrix[i, j]
                G.add_edge(headers[i], headers[j], weight=weight)

    if weighted_edgelist_file:
        nx.write_weighted_edgelist(G, weighted_edgelist_file)

    if output_file:
        nx.write_adjlist(G, output_file)
    else:
        nx.write_adjlist(G, sys.stdout)

def compute_causal_matrix(data: np.ndarray, max_lags: int) -> np.ndarray:
    """Calculate the summary matrix using DirectLiNGAM for time series data."""
    # Time VAR procedure
    var_start = time.time()
    var = VAR(data)
    result = var.fit(maxlags=max_lags, trend="n")
    residuals = result.resid
    var_time = time.time() - var_start
    print(f'VAR time: {var_time:.6f} seconds')
    
    # Time DirectLiNGAM procedure
    lingam_start = time.time()
    causal_order = analyze_direct_lingam(residuals)
    
    # Compute adjacency matrix (only once now)
    B = np.zeros([residuals.shape[1], residuals.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]
        if len(predictors) > 0:
            B[target, predictors] = compute_adaptive_lasso(residuals, predictors, target)
    
    lingam_time = time.time() - lingam_start
    print(f'DirectLiNGAM time: {lingam_time:.6f} seconds')
    
    summary_matrix = np.abs(B).transpose()
    np.fill_diagonal(summary_matrix, 0)
    return summary_matrix

def analyze_direct_lingam(X):
    """Fit DirectLiNGAM to the data."""
    X = check_array(X)
    n_features = X.shape[1]
    U = set(range(n_features))  # Use set for O(1) removal
    K = []
    
    # Standardize X once
    X_ = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Compute these once outside the loop
    entropies = compute_variable_entropies(X_)
    residual_entropies = compute_residual_entropies(X_)

    while U:  # Simplified loop condition
        M_array = np.zeros(len(U))
        U_list = list(U)  # Convert to list for indexing
        
        for idx, i in enumerate(U_list):
            # Vectorized computation of M
            j_values = np.array([j for j in U_list if j != i])
            if len(j_values) > 0:
                diffs = np.array([compute_mutual_info_diff(entropies, residual_entropies, i, j) 
                                for j in j_values])
                M_array[idx] = -np.sum(np.minimum(0, diffs) ** 2)
        
        m_idx = np.argmax(M_array)
        m = U_list[m_idx]
        K.append(m)
        U.remove(m)
    
    return K

def compute_variable_entropies(X_std):
    """Calculate entropies for each variable."""
    k1, k2, gamma = 79.047, 7.4129, 0.37457
    log_cosh = np.log(np.cosh(X_std))
    exp_term = X_std * np.exp(-(X_std ** 2) / 2)
    return (1 + np.log(2 * np.pi)) / 2 - k1 * (np.mean(log_cosh, axis=0) - gamma) ** 2 - k2 * (np.mean(exp_term, axis=0)) ** 2

def compute_residual_entropies(X_):
    """Calculate residual entropies between pairs of variables using vectorized operations."""
    n_features = X_.shape[1]
    variances = np.var(X_, axis=0)
    covariances = np.cov(X_.T, bias=True)
    X_entropy = np.zeros((n_features, n_features))
    k1, k2, gamma = 79.047, 7.4129, 0.37457
    
    # Compute all betas at once
    betas = covariances / variances[None, :]
    np.fill_diagonal(betas, 0)  # Avoid division by zero
    
    # Vectorized computation for all pairs
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                residual = X_[:, i] - betas[i, j] * X_[:, j]
                residual_std = residual / np.std(residual)
                
                log_cosh_mean = np.mean(np.log(np.cosh(residual_std)))
                exp_term_mean = np.mean(residual_std * np.exp(-(residual_std ** 2) / 2))
                
                X_entropy[i, j] = ((1 + np.log(2 * np.pi)) / 2 - 
                                 k1 * (log_cosh_mean - gamma) ** 2 - 
                                 k2 * exp_term_mean ** 2)
    
    return X_entropy

def compute_mutual_info_diff(entropies, residual_entropies, i, j):
    """Calculate the difference in mutual information."""
    return (entropies[j] + residual_entropies[i][j]) - (entropies[i] + residual_entropies[j][i])

def compute_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict using adaptive lasso regression."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_std[:, predictors], X_std[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X_std[:, predictors] * weight, X_std[:, target])
    pruned_idx = np.abs(reg.coef_ * weight) > 0.0
    coef = np.zeros(reg.coef_.shape)
    if pruned_idx.sum() > 0:
        lr = LinearRegression()
        pred = np.array(predictors)
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_
    return coef

def compute_b_matrices(X, B0, M_taus):
    """Calculate B matrices from VAR coefficients and DirectLiNGAM adjacency matrix."""
    n_features = X.shape[1]
    B_taus = np.array([B0])
    for M in M_taus:
        B_taus = np.append(B_taus, [np.dot((np.eye(n_features) - B0), M)], axis=0)
    return B_taus

if __name__ == '__main__':
    main()
