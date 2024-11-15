#!/usr/bin/env python3

import argparse
import numpy as np
import networkx as nx
import sys
import time
import lingam

def calculate_summary_matrix(data: np.ndarray, max_lags: int) -> np.ndarray:
    """
    Calculate the summary matrix using FastLiNGAM for time series data.
    
    Args:
        data: Input time series data
        max_lags: Maximum number of lags to consider
        
    Returns:
        summary_matrix: Matrix of causal relationships
    """
    model = lingam.VARLiNGAM(lags=max_lags, prune=False)
    model.fit(data)
    
    # Get the first adjacency matrix (contemporary relationships)
    summary_matrix = np.abs(model.adjacency_matrices_[0]).transpose()
    np.fill_diagonal(summary_matrix, 0)  # Remove self-loops
    
    return summary_matrix

def process_csv(input_file, output_file, time_file, weighted_edgelist_file, max_lags, weight_threshold):
    # Read data
    if input_file == 'stdin':
        raw_data = np.genfromtxt(sys.stdin, delimiter=',', skip_header=1)
        headers = next(sys.stdin).strip().split(',')  # Read header for variable names
    else:
        raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
        headers = np.genfromtxt(input_file, delimiter=',', max_rows=1, dtype=str)
    
    # Process data (calculate returns)
    data = (raw_data[1:] / raw_data[:-1]) - 1
    
    # Start timing
    start_time = time.time()
    
    # Calculate summary matrix
    summary_matrix = calculate_summary_matrix(data, max_lags)
    
    # End timing
    computation_time = time.time() - start_time
    
    # Write computation time if time_file is specified
    if time_file:
        with open(time_file, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Create a directed graph from the summary matrix
    G = nx.DiGraph()
    
    # First add all nodes to ensure isolated nodes are included
    for header in headers:
        G.add_node(header)
    
    # Then add edges that meet the weight threshold
    for i in range(len(headers)):
        for j in range(len(headers)):
            if summary_matrix[i, j] > weight_threshold:
                weight = summary_matrix[i, j]
                G.add_edge(headers[i], headers[j], weight=weight)

    # Export weighted edgelist if requested
    if weighted_edgelist_file:
        nx.write_weighted_edgelist(G, weighted_edgelist_file)

    # Export the graph (regular adjacency list)
    if output_file:
        nx.write_adjlist(G, output_file)
    else:
        nx.write_adjlist(G, sys.stdout)

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
    args = parser.parse_args()

    process_csv(args.input, args.output, args.time, args.weighted_edgelist, args.max_lags, args.weight_threshold)

if __name__ == '__main__':
    main() 
