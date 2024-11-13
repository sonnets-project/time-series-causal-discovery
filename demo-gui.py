import tkinter as tk
import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox
from tkinter import ttk

class CSVViewer:
    def __init__(self, master):
        self.pos = None
        self.master = master
        self.master.title("Causal relationships in S&P 100 (400-day window)")
        
        # Create a main frame to hold all widgets
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=1)
        
        # Create left frame for the graph (62% of window width)
        self.left_frame = tk.Frame(self.main_frame, width=self.master.winfo_screenwidth() * 0.62)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Create right frame (38% of window width)
        self.right_frame = tk.Frame(self.main_frame, width=self.master.winfo_screenwidth() * 0.38)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        
        # Split right frame into three parts with adjusted proportions
        self.right_upper_frame = tk.Frame(self.right_frame, height=self.master.winfo_screenheight() * 0.27)  # 27% of height
        self.right_upper_frame.pack(side=tk.TOP, fill=tk.BOTH)
        self.right_upper_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        self.right_middle_frame = tk.Frame(self.right_frame, height=self.master.winfo_screenheight() * 0.27)  # 27% of height
        self.right_middle_frame.pack(side=tk.TOP, fill=tk.BOTH)
        self.right_middle_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        self.right_bottom_frame = tk.Frame(self.right_frame)  # Takes remaining space
        self.right_bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        
        # Upper frame - Most influential nodes caption with larger font
        self.upper_caption = tk.Label(self.right_upper_frame, 
                                    text="Most influential nodes (Top 10)", 
                                    font=('Arial', 18, 'bold'))
        self.upper_caption.pack(side=tk.TOP, pady=5)
        
        self.tree_out = ttk.Treeview(self.right_upper_frame, columns=('Node', 'Effects'), show='headings')
        self.tree_out.heading('Node', text='Node')
        self.tree_out.heading('Effects', text='Effects')
        self.tree_out.column('Node', width=40)
        self.tree_out.pack(fill=tk.BOTH, expand=1)
        
        # Middle frame - Least influential nodes caption with larger font
        self.middle_caption = tk.Label(self.right_middle_frame, 
                                     text="Least influential nodes (Bottom 10)", 
                                     font=('Arial', 18, 'bold'))
        self.middle_caption.pack(side=tk.TOP, pady=5)
        
        self.tree_least = ttk.Treeview(self.right_middle_frame, columns=('Node', 'Effects'), show='headings')
        self.tree_least.heading('Node', text='Node')
        self.tree_least.heading('Effects', text='Effects')
        self.tree_least.column('Node', width=40)
        self.tree_least.pack(fill=tk.BOTH, expand=1)
        
        # Bottom frame - Edge changes and statistics with larger font
        self.edge_changes_label = tk.Label(self.right_bottom_frame, 
                                         text="Causal Graph Statistics:", 
                                         font=('Arial', 18, 'bold'))
        self.edge_changes_label.pack(side=tk.TOP, pady=15)
        
        self.new_edges_label = tk.Label(self.right_bottom_frame, 
                                      text="New Edges: 0", 
                                      font=('Arial', 15))
        self.new_edges_label.pack(side=tk.TOP, pady=8)
        
        self.removed_edges_label = tk.Label(self.right_bottom_frame, 
                                          text="Removed Edges: 0", 
                                          font=('Arial', 15))
        self.removed_edges_label.pack(side=tk.TOP, pady=8)
        
        self.avg_out_edges_label = tk.Label(self.right_bottom_frame, 
                                          text="Average Outgoing Edges: 0.00", 
                                          font=('Arial', 15))
        self.avg_out_edges_label.pack(side=tk.TOP, pady=8)
        
        self.avg_in_edges_label = tk.Label(self.right_bottom_frame, 
                                         text="Average Incoming Edges: 0.00", 
                                         font=('Arial', 15))
        self.avg_in_edges_label.pack(side=tk.TOP, pady=8)

        # Store previous frame's edges
        self.previous_edges = set()

        # Initialize the figure attribute
        self.figure = Figure()  # Adjust figure size as needed

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.frame_id_label = tk.Label(self.left_frame, text="Frame ID: ")
        self.frame_id_label.pack(side='top')
        
        self.folder_path = os.path.join(os.getcwd(), 'matrices')
        self.csv_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.csv')])
        self.current_index = 700  # Start from the first frame
        self.top_nodes_history = {}  # Store top nodes for each frame
        
        # Read the header line from 'sp100.csv' to get node labels
        self.node_labels = []
        try:
            sp100_path = os.path.join(os.getcwd(), 'data', 'sp100.csv')
            with open(sp100_path, 'r') as f:
                self.node_labels = f.readline().strip().split(',')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read node labels from 'sp100.csv'\n{e}")
            self.node_labels = []

        if not self.csv_files:
            messagebox.showerror("Error", "No CSV files found in the 'matrices' folder")
        else:
            self.load_frame(self.current_index)
        
        # Bind keyboard shortcuts
        self.master.bind('<Left>', lambda event: self.load_prev_frame())
        self.master.bind('<Right>', lambda event: self.load_next_frame())
        
        # Display keyboard shortcuts
        self.shortcut_label = tk.Label(self.left_frame, text="Shortcuts: Left Arrow - Previous Frame, Right Arrow - Next Frame", bg='white', fg='black')
        self.shortcut_label.pack(side='bottom')

    def load_frame(self, index):
        file_path = os.path.join(self.folder_path, self.csv_files[index])
        
        try:
            array = pd.read_csv(file_path, header=None).values.T
            # quantile_value = np.quantile(array, 0.99)
            quantile_value = 0.43
            array[array < quantile_value] = 0
            G = nx.DiGraph(array)
            if self.pos is None:

                self.pos = nx.spring_layout(G, k=0.5)
            pos = self.pos
            
            self.figure.clf()
            
            fig = self.figure
            ax = fig.add_subplot(111)

            # Identify top 10 most influential nodes for the current frame
            top_nodes = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:10]
            current_top_nodes = set(node for node, _ in top_nodes)
            self.top_nodes_history[index] = current_top_nodes

            # Get previous frame's top nodes
            previous_top_nodes = self.top_nodes_history.get(index - 1, set())

            # Prepare node colors and edge colors
            node_colors = []
            node_edge_colors = []
            for i in range(len(G.nodes)):
                if i in current_top_nodes:
                    if i in previous_top_nodes:
                        node_colors.append('white')
                        node_edge_colors.append('red')
                    else:
                        node_colors.append('red')
                        node_edge_colors.append('red')
                else:
                    node_colors.append('white')
                    node_edge_colors.append('black')

            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_colors = [plt.cm.Blues(weight / max_weight) for weight in edge_weights]

            nx.draw(G, pos, ax=ax, with_labels=True, node_size=300, font_size=8, 
                    edge_color=edge_colors, arrows=True, labels=dict(enumerate(self.node_labels)), 
                    node_color=node_colors, edgecolors=node_edge_colors, linewidths=2)
            
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            self.canvas.figure = fig
            self.canvas.draw()
            
            self.frame_id_label.config(text=f"Frame ID: {index + 1}")
            
            self.update_tables(G)
            
            # Track edge changes
            current_edges = set(G.edges())
            new_edges = len(current_edges - self.previous_edges)
            removed_edges = len(self.previous_edges - current_edges)
            
            # Calculate average edges
            out_degrees = [d for n, d in G.out_degree()]
            in_degrees = [d for n, d in G.in_degree()]
            avg_out = sum(out_degrees) / len(out_degrees) if out_degrees else 0
            avg_in = sum(in_degrees) / len(in_degrees) if in_degrees else 0
            
            # Update labels
            self.new_edges_label.config(text=f"New Edges: {new_edges}")
            self.removed_edges_label.config(text=f"Removed Edges: {removed_edges}")
            self.avg_out_edges_label.config(text=f"Average Outgoing Edges: {avg_out:.2f}")
            self.avg_in_edges_label.config(text=f"Average Incoming Edges: {avg_in:.2f}")
            
            # Store current edges for next comparison
            self.previous_edges = current_edges
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load and process CSV file\n{e}")
    
    def update_tables(self, G, top_n=10):
        # Clear existing items
        for tree in [self.tree_out, self.tree_least]:
            for i in tree.get_children():
                tree.delete(i)
        
        # Get nodes with outgoing edges, considering average weight
        out_degree_with_avg = [(node, degree, np.mean([G[node][succ]['weight'] for succ in G.successors(node)])) 
                               for node, degree in G.out_degree()]
        
        # Top influential nodes
        top_nodes_out = sorted(out_degree_with_avg, key=lambda x: (x[1], x[2]), reverse=True)[:top_n]
        
        # Least influential nodes
        least_nodes_out = sorted(out_degree_with_avg, key=lambda x: (x[1], x[2]))[:top_n]
        
        # Update most influential nodes table
        for node, _, _ in top_nodes_out:
            node_label = self.node_labels[node]
            effects = sorted(G.successors(node), key=lambda x: G[node][x]['weight'], reverse=True)
            effect_labels = ', '.join([self.node_labels[e] for e in effects])
            self.tree_out.insert('', 'end', values=(node_label, effect_labels))
        
        # Update least influential nodes table
        for node, _, _ in least_nodes_out:
            node_label = self.node_labels[node]
            effects = sorted(G.successors(node), key=lambda x: G[node][x]['weight'], reverse=True)
            effect_labels = ', '.join([self.node_labels[e] for e in effects])
            self.tree_least.insert('', 'end', values=(node_label, effect_labels))

    def load_prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_frame(self.current_index)
    
    def load_next_frame(self):
        if self.current_index < len(self.csv_files) - 1:
            self.current_index += 1
            self.load_frame(self.current_index)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")
    app = CSVViewer(root)
    root.mainloop()
    

