#!/usr/bin/env python3

import argparse
import numpy as np
from common import granger_causality_matrix  # Import the function
import os
import networkx as nx
import sys
import time

def process_csv(input_file, output_file, time_file, weighted_edgelist_file, max_lags, significance_level, metric):
    # Read data
    raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    headers = np.genfromtxt(input_file, delimiter=',', max_rows=1, dtype=str)
    
    # Start timing
    start_time = time.time()
    
    # Compute Granger causality matrix
    causality_matrix, detailed_results = granger_causality_matrix(
        data=raw_data,
        max_lags=max_lags,
        significance_level=significance_level,
        metric=metric
    )
    
    # End timing
    computation_time = time.time() - start_time
    
    # Write computation time if time_file is specified
    if time_file:
        with open(time_file, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Create a directed graph from the causality matrix
    G = nx.DiGraph()
    for i, row in enumerate(causality_matrix):
        for j, value in enumerate(row):
            if value:  # If there is a causal relationship
                # Get the maximum F-statistic across all lags as the weight
                _, _, f_stats = detailed_results[(i, j)]
                weight = max(f_stats.values()) if f_stats else 0.0
                G.add_edge(headers[i], headers[j], weight=weight)

    # Export weighted edgelist if requested
    if weighted_edgelist_file:
        nx.write_weighted_edgelist(G, weighted_edgelist_file)

    # Export the graph (regular adjacency list)
    nx.write_adjlist(G, output_file)

def main():
    parser = argparse.ArgumentParser(description='Discover Granger causality relationships in time series data.')
    parser.add_argument('-i', '--input', required=True,
                        help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file path')
    parser.add_argument('-t', '--time', default=None,
                        help='Output file for computation time in seconds')
    parser.add_argument('--max_lags', type=int, default=5,
                        help='Maximum number of lags to test (default: 5)')
    parser.add_argument('--significance_level', type=float, default=0.05,
                        help='Significance level for the test (default: 0.05)')
    parser.add_argument('--metric', type=str, default='is_causal',
                        choices=['is_causal', 'min_pvalue', 'max_fstat'],
                        help='Metric to use for the result matrix (default: is_causal)')
    parser.add_argument('-w', '--weighted_edgelist', default=None,
                        help='Output file for f-statistic weighted edgelist')
    
    args = parser.parse_args()
    
    process_csv(args.input, 
                output_file=args.output,
                time_file=args.time,
                weighted_edgelist_file=args.weighted_edgelist,
                max_lags=args.max_lags, 
                significance_level=args.significance_level, 
                metric=args.metric)

if __name__ == "__main__":
    main()
