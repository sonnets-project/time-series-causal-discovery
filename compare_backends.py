import sys
import os
import subprocess
import networkx as nx
import numpy as np
from pathlib import Path

def run_backend(backend_name, data_path):
    """Run a specific backend on the given data file."""
    backend_dir = Path('backend') / backend_name
    csvif_path = backend_dir / 'csvif.py'
    output_path = Path('causal_graphs') / f'{backend_name}_result.txt'
    time_path = Path('causal_graphs') / f'{backend_name}_time.txt'
    
    # Ensure the causal_graphs directory exists
    os.makedirs('causal_graphs', exist_ok=True)
    
    # Run the backend using csvif.py interface with appropriate arguments
    try:
        subprocess.run([
            'python3',
            str(csvif_path),
            '-i', str(data_path),
            '-o', str(output_path),
            '-t', str(time_path)
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error running {backend_name}: {e}")
        return None

def load_graph_from_file(file_path):
    """Load a graph from the output file."""
    G = nx.DiGraph()
    
    try:
        print(f"\nAttempting to load graph from: {file_path}")
        
        if not file_path or not Path(file_path).exists():
            print(f"File does not exist: {file_path}")
            return None
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Process adjacency list format
        # Each line format: "node neighbor1 neighbor2 neighbor3 ..."
        nodes = set()
        edges = []
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
                
            parts = line.strip().split()
            if not parts:
                continue
                
            source = parts[0]
            nodes.add(source)
            
            # Add edges to all neighbors listed on this line
            for target in parts[1:]:
                nodes.add(target)
                edges.append((source, target))
        
        print(f"Found {len(nodes)} nodes")
        
        # Add nodes to graph
        G.add_nodes_from(nodes)
        
        # Add edges
        G.add_edges_from(edges)
        print(f"Added {len(edges)} edges")
        
        return G
    except Exception as e:
        print(f"Error loading graph from {file_path}: {str(e)}")
        return None

def structural_hamming_distance(G1, G2):
    """
    Calculate the Structural Hamming Distance between two graphs.
    SHD counts the number of edge insertions, deletions, or flips needed to transform one graph into another.
    """
    if not (G1 and G2):
        return None
    
    # Ensure both graphs have the same nodes
    if set(G1.nodes()) != set(G2.nodes()):
        raise ValueError("Graphs must have the same nodes")
    
    distance = 0
    for node1 in G1.nodes():
        for node2 in G1.nodes():
            if node1 == node2:
                continue
                
            edge1 = G1.has_edge(node1, node2)
            edge1_rev = G1.has_edge(node2, node1)
            edge2 = G2.has_edge(node1, node2)
            edge2_rev = G2.has_edge(node2, node1)
            
            # Different edge types (missing, extra, or reversed)
            if (edge1 and not edge2) or (edge2 and not edge1) or \
               (edge1_rev and not edge2_rev) or (edge2_rev and not edge1_rev):
                distance += 1
    
    # Divide by 2 because each difference was counted twice
    return distance // 2

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_backends.py <data_file.csv>")
        sys.exit(1)
    
    data_path = Path(sys.argv[1])
    if not data_path.exists():
        print(f"Error: Data file {data_path} does not exist")
        sys.exit(1)
    
    # Run both backends
    print("Running FastLiNGAM...")
    fastlingam_output = run_backend('fastlingam', data_path)
    print(f"FastLiNGAM output file: {fastlingam_output}")
    
    print("\nRunning MiniLiNGAM...")
    minilingam_output = run_backend('minilingam', data_path)
    print(f"MiniLiNGAM output file: {minilingam_output}")
    
    # Load the resulting graphs
    print("\nLoading graphs...")
    fastlingam_graph = load_graph_from_file(fastlingam_output)
    minilingam_graph = load_graph_from_file(minilingam_output)
    
    # Compare the graphs
    if fastlingam_graph and minilingam_graph:
        shd = structural_hamming_distance(fastlingam_graph, minilingam_graph)
        
        print("\nComparison Results:")
        print("-" * 50)
        print(f"FastLiNGAM nodes: {len(fastlingam_graph.nodes())}")
        print(f"FastLiNGAM edges: {len(fastlingam_graph.edges())}")
        print(f"MiniLiNGAM nodes: {len(minilingam_graph.nodes())}")
        print(f"MiniLiNGAM edges: {len(minilingam_graph.edges())}")
        print(f"Structural Hamming Distance: {shd}")
        
        # Print unique edges in each graph
        print("\nEdges unique to FastLiNGAM:")
        for edge in fastlingam_graph.edges():
            if not minilingam_graph.has_edge(*edge):
                print(f"{edge[0]} -> {edge[1]}")
        
        print("\nEdges unique to MiniLiNGAM:")
        for edge in minilingam_graph.edges():
            if not fastlingam_graph.has_edge(*edge):
                print(f"{edge[0]} -> {edge[1]}")
                
        # Clean up the output files
        if fastlingam_output and Path(fastlingam_output).exists():
            Path(fastlingam_output).unlink()
        if minilingam_output and Path(minilingam_output).exists():
            Path(minilingam_output).unlink()
    else:
        print("Error: Failed to load one or both graphs")

if __name__ == "__main__":
    main() 