import networkx as nx
import numpy as np
import os
import lingam
import sys

def causal_discovery_varlingam(data, lag=1):
    # Run VARLiNGAM with a default lag of 1
    model = lingam.VARLiNGAM(lags=lag)
    model.fit(data)
    summary_matrix = np.sum(np.abs(model.adjacency_matrices_), axis=0)
    causal_graph = nx.from_numpy_array(summary_matrix.T, create_using=nx.DiGraph)
    for u, v, d in causal_graph.edges(data=True):
        del d['weight']
    return causal_graph

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_causal_discovery.py data_file')
        exit()
        
    output_directory = "./causal_graphs"
    os.makedirs(output_directory, exist_ok=True)

    data_filename = sys.argv[1]

    # Load data from the specified CSV file, skipping the header
    data = np.genfromtxt(data_filename, delimiter=',', skip_header=1)
    print(f'Running causal discovery on {data_filename}')
    
    # Run the VARLiNGAM causal discovery
    G = causal_discovery_varlingam(data)
    
    # Save the causal graph as an adjacency list
    output_path = os.path.join(output_directory, 'causal_graph_adj_list.txt')
    with open(output_path, 'w') as f:
        for line in nx.generate_adjlist(G):
            f.write(f"{line}\n")
    print(f'Causal graph saved as adjacency list to {output_path}')
    