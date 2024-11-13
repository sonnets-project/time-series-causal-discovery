import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import product
from typing import Tuple, Optional, Dict

def granger_causality_matrix(
    data: np.ndarray,
    max_lags: int,
    significance_level: float = 0.05,
    metric: str = 'is_causal'
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Compute pairwise Granger causality matrix for multiple time series.
    """
    def _single_granger_test(x: np.ndarray, y: np.ndarray) -> tuple:
        """Compute Granger Causality between two 1D time series."""
        data = np.column_stack((y, x))
        test_results = grangercausalitytests(data, maxlag=max_lags, verbose=False)
        
        p_values = {}
        f_stats = {}
        for lag in range(1, max_lags + 1):
            p_values[lag] = test_results[lag][0]['ssr_chi2test'][1]
            f_stats[lag] = test_results[lag][0].get('ssr_f', (None,))[0]
        
        is_causal = any(p_value < significance_level for p_value in p_values.values())
        return is_causal, p_values, f_stats

    # Input validation
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    
    n_series = data.shape[1]
    pairs = list(product(range(n_series), repeat=2))
    
    # Initialize result matrix and detailed results dictionary
    result_matrix = np.zeros((n_series, n_series))
    detailed_results = {(i, j): None for i, j in pairs}
    
    # Compute results in a single-threaded manner
    for i, j in pairs:
        if i == j:
            detailed_results[(i, j)] = (False, {}, {})
            continue
        detailed_results[(i, j)] = _single_granger_test(data[:, i], data[:, j])
        
        is_causal, p_values, f_stats = detailed_results[(i, j)]
        if metric == 'is_causal':
            result_matrix[i, j] = int(is_causal)
        elif metric == 'min_pvalue':
            result_matrix[i, j] = min(p_values.values()) if p_values else 1.0
        elif metric == 'max_fstat':
            result_matrix[i, j] = max(f_stats.values()) if f_stats else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return result_matrix, detailed_results
