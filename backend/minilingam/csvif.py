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
    parser.add_argument('--measure', type=str, default='pwling_v2', choices=['pwling', 'pwling_v2', 'pwling_v3'],
                      help='LiNGAM measure to use (default: pwling_v2)')
    args = parser.parse_args()
    analyze_csv_data(args.input, args.output, args.time, args.weighted_edgelist, 
                    args.max_lags, args.weight_threshold, args.measure)

def analyze_csv_data(input_file, output_file, time_file, weighted_edgelist_file, max_lags, weight_threshold, measure):
    if input_file == 'stdin':
        raw_data = np.genfromtxt(sys.stdin, delimiter=',', skip_header=1)
        headers = next(sys.stdin).strip().split(',')
    else:
        raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
        headers = np.genfromtxt(input_file, delimiter=',', max_rows=1, dtype=str)
    
    data = (raw_data[1:] / raw_data[:-1]) - 1
    start_time = time.time()
    summary_matrix = compute_causal_matrix(data, max_lags, measure)
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

def compute_causal_matrix(data: np.ndarray, max_lags: int, measure: str) -> np.ndarray:
    """Calculate the summary matrix using DirectLiNGAM for time series data."""
    var = VAR(data)
    result = var.fit(maxlags=max_lags, trend="n")
    residuals = result.resid
    
    causal_order, adjacency_matrix = analyze_direct_lingam(residuals, measure=measure)
    print('Causal order:', causal_order)
    
    summary_matrix = np.abs(adjacency_matrix).transpose()
    np.fill_diagonal(summary_matrix, 0)
    return summary_matrix

def analyze_direct_lingam(X, measure="pwling"):
    """Fit DirectLiNGAM to the data."""
    X = check_array(X)
    n_features = X.shape[1]
    U = np.arange(n_features)
    K = []
    
    X_ = np.copy(X)
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
    
    entropies = compute_variable_entropies(X_)
    residual_entropies = compute_residual_entropies(X_)

    for _ in range(n_features):
        M_list = []
        for i in U:
            M = 0
            for j in U:
                if i != j:
                    M += min([0, compute_mutual_info_diff(entropies, residual_entropies, i, j)]) ** 2
            M_list.append(-M)
        m = U[np.argmax(M_list)]
        K.append(m)
        U = U[U != m]

    B = compute_adjacency_matrix(X, K)
    return K, B

def analyze_var_lingam(X, lags=1, criterion="bic", prune=True, lingam_measure='pwling'):
    """Main function that combines VAR and LiNGAM analysis."""
    M_taus, lags, residuals = estimate_var_coefficients(X, lags, criterion)
    causal_order, adjacency_matrix = analyze_direct_lingam(residuals, measure=lingam_measure)
    B_taus = compute_b_matrices(X, adjacency_matrix, M_taus)
    if prune:
        B_taus = prune_matrices(X, B_taus, causal_order, lags)
    return causal_order, B_taus, residuals

def compute_variable_entropies(X_std):
    """Calculate entropies for each variable."""
    k1, k2, gamma = 79.047, 7.4129, 0.37457
    log_cosh = np.log(np.cosh(X_std))
    exp_term = X_std * np.exp(-(X_std ** 2) / 2)
    return (1 + np.log(2 * np.pi)) / 2 - k1 * (np.mean(log_cosh, axis=0) - gamma) ** 2 - k2 * (np.mean(exp_term, axis=0)) ** 2

def compute_residual_entropies(X_):
    """Calculate residual entropies between pairs of variables."""
    n_features = X_.shape[1]
    variances = np.var(X_, axis=0)
    covariances = np.cov(X_.T, bias=True)
    X_entropy = np.zeros((n_features, n_features))
    k1, k2, gamma = 79.047, 7.4129, 0.37457
    
    for i, j in itertools.product(range(n_features), range(n_features)):
        if i != j:
            beta = covariances[i, j] / variances[j]
            residual_std = (X_[:, i] - beta * X_[:, j]) / np.std(X_[:, i] - beta * X_[:, j])
            log_cosh_mean = np.mean(np.log(np.cosh(residual_std)))
            exp_term_mean = np.mean(residual_std * np.exp(-(residual_std ** 2) / 2))
            X_entropy[i, j] = (1 + np.log(2 * np.pi)) / 2 - k1 * (log_cosh_mean - gamma) ** 2 - k2 * exp_term_mean ** 2
    return X_entropy

def compute_mutual_info_diff(entropies, residual_entropies, i, j):
    """Calculate the difference in mutual information."""
    return (entropies[j] + residual_entropies[i][j]) - (entropies[i] + residual_entropies[j][i])

def estimate_var_coefficients(X, lags=1, criterion="bic"):
    """Estimate VAR coefficients using the specified criterion."""
    var = VAR(X)
    if criterion not in ["aic", "fpe", "hqic", "bic"]:
        result = var.fit(maxlags=lags, trend="n")
        return result.coefs, result.k_ar, result.resid
    
    min_value = float("Inf")
    result = None
    for lag in range(1, lags + 1):
        fitted = var.fit(maxlags=lag, ic=None, trend="n")
        value = getattr(fitted, criterion)
        if value < min_value:
            min_value, result = value, fitted
    return result.coefs, result.k_ar, result.resid

def compute_b_matrices(X, B0, M_taus):
    """Calculate B matrices from VAR coefficients and DirectLiNGAM adjacency matrix."""
    n_features = X.shape[1]
    B_taus = np.array([B0])
    for M in M_taus:
        B_taus = np.append(B_taus, [np.dot((np.eye(n_features) - B0), M)], axis=0)
    return B_taus

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

def compute_adjacency_matrix(X, causal_order):
    """Estimate the adjacency matrix using adaptive lasso."""
    B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]
        if len(predictors) > 0:
            B[target, predictors] = compute_adaptive_lasso(X, predictors, target)
    return B

def prune_matrices(X, B_taus, causal_order, lags):
    """Prune the B matrices using adaptive lasso."""
    n_features = X.shape[1]
    stacked = [np.flip(X, axis=0)]
    for i in range(lags):
        stacked.append(np.roll(stacked[-1], -1, axis=0))
    blocks = np.array(list(zip(*stacked)))[: -lags]
    for i in range(n_features):
        causal_order_no = causal_order.index(i)
        ancestor_indexes = causal_order[:causal_order_no]
        obj = np.zeros((len(blocks)))
        exp = np.zeros((len(blocks), causal_order_no + n_features * lags))
        for j, block in enumerate(blocks):
            obj[j] = block[0][i]
            exp[j:] = np.concatenate([block[0][ancestor_indexes].flatten(), block[1:][:].flatten()], axis=0)
        predictors = list(range(exp.shape[1]))
        target = len(predictors)
        X_con = np.concatenate([exp, obj.reshape(-1, 1)], axis=1)
        coef = compute_adaptive_lasso(X_con, predictors, target)
        B_taus[0][i, ancestor_indexes] = coef[:causal_order_no]
        for j in range(len(B_taus[1:])):
            B_taus[j + 1][i, :] = coef[causal_order_no + n_features * j:causal_order_no + n_features * j + n_features]
    return B_taus

if __name__ == '__main__':
    main()
