import argparse
import numpy as np
import networkx as nx
import lingam
import os

def calculate_summary_matrix(data):
    model = lingam.VARLiNGAM(lags=2, prune=False)
    model.fit(data)
    summary_matrix = np.abs(model.adjacency_matrices_[0])
    np.fill_diagonal(summary_matrix, 0)
    return summary_matrix

def process_csv(input_file, output_file, window_size=500, percentile_threshold=99):
    # Read data
    raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    with open(input_file, 'r') as f:
        variable_names = f.readline().strip().split(',')
    
    # Process data (using returns calculation as in your notebook)
    data = (raw_data[1:] / raw_data[:-1]) - 1
    
    # Calculate summary matrix
    summary_matrix = calculate_summary_matrix(data)
    
    # Create graph based on percentile threshold
    percentile = np.percentile(summary_matrix, percentile_threshold)
    summary_matrix_filtered = summary_matrix > percentile
    
    # Create and save graph
    G = nx.from_numpy_array(summary_matrix_filtered.T, create_using=nx.DiGraph)
    mapping = {i: variable_names[i] for i in range(len(variable_names))}
    G = nx.relabel_nodes(G, mapping)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as adjacency list
    nx.write_adjlist(G, output_file)

def main():
    parser = argparse.ArgumentParser(description='Discover causal relationships in multivariate time series data.')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output adjacency list file path')
    parser.add_argument('--window', type=int, default=500, help='Window size for analysis (default: 500)')
    parser.add_argument('--percentile', type=float, default=99, 
                        help='Percentile threshold for edge filtering (default: 99)')
    
    args = parser.parse_args()
    
    process_csv(args.input_file, args.output_file, 
               window_size=args.window, 
               percentile_threshold=args.percentile)

if __name__ == "__main__":
    main() 
