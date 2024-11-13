import argparse
import numpy as np
from common import granger_causality_matrix  # Import the function
import os
import networkx as nx  # Import NetworkX
import sys

def process_csv(input_file, output_file=None, time_file=None, max_lags=5, significance_level=0.05, metric='is_causal'):
    # Read data
    raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    headers = np.genfromtxt(input_file, delimiter=',', max_rows=1, dtype=str)  # Read header for variable names
    
    # Start timing
    import time
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
                G.add_edge(headers[i], headers[j])

    # Export the graph
    if output_file:
        nx.write_adjlist(G, output_file)
    else:
        # If no output file is specified, write to stdout
        nx.write_adjlist(G, sys.stdout)

def main():
    parser = argparse.ArgumentParser(description='Discover Granger causality relationships in time series data.')
    parser.add_argument('-i', '--input', help='Input CSV file path (default: standard input)', default=None)
    parser.add_argument('-o', '--output', help='Output file path (default: standard output)', default=None)
    parser.add_argument('-t', '--time', help='Output file for computation time in seconds', default=None)
    parser.add_argument('--max_lags', type=int, default=5, help='Maximum number of lags to test (default: 5)')
    parser.add_argument('--significance_level', type=float, default=0.05, help='Significance level for the test (default: 0.05)')
    parser.add_argument('--metric', type=str, default='is_causal', choices=['is_causal', 'min_pvalue', 'max_fstat'],
                        help='Metric to use for the result matrix (default: is_causal)')
    
    args = parser.parse_args()
    
    input_file = args.input if args.input else 'stdin'  # Use 'stdin' if no input file is provided
    output_file = args.output  # Use provided output file or None for standard output
    process_csv(input_file, 
                 output_file=output_file,
                 time_file=args.time,
                 max_lags=args.max_lags, 
                 significance_level=args.significance_level, 
                 metric=args.metric)

if __name__ == "__main__":
    main()
