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
        self.master.title("Causal relationships in S&P 100 (500-day window)")
        
        # Create a main frame to hold all widgets
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=1)
        
        # Create left frame for the graph (62% of window width)
        self.left_frame = tk.Frame(self.main_frame, width=self.master.winfo_screenwidth() * 0.62)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Create right frame (38% of window width)
        self.right_frame = tk.Frame(self.main_frame, width=self.master.winfo_screenwidth() * 0.38)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        
        # Split right frame into upper and lower frames
        self.right_upper_frame = tk.Frame(self.right_frame)
        self.right_upper_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.right_lower_frame = tk.Frame(self.right_frame)
        self.right_lower_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        
        # Add caption for the upper table
        self.upper_caption = tk.Label(self.right_upper_frame, text="Most influential nodes")
        self.upper_caption.pack(side=tk.TOP)
        
        # Create table in the upper right frame for outgoing edges
        self.tree_out = ttk.Treeview(self.right_upper_frame, columns=('Node', 'Effects'), show='headings')
        self.tree_out.heading('Node', text='Node')
        self.tree_out.heading('Effects', text='Effects')
        self.tree_out.column('Node', width=40)  # Adjust width as needed
        self.tree_out.pack(fill=tk.BOTH, expand=1)
        
        # Add caption for the lower table
        self.lower_caption = tk.Label(self.right_lower_frame, text="Most submissive nodes")
        self.lower_caption.pack(side=tk.TOP)
        
        # Create table in the lower right frame for incoming edges
        self.tree_in = ttk.Treeview(self.right_lower_frame, columns=('Node', 'Causes'), show='headings')
        self.tree_in.heading('Node', text='Node')
        self.tree_in.heading('Causes', text='Causes')
        self.tree_in.column('Node', width=40)  # Adjust width as needed
        self.tree_in.pack(fill=tk.BOTH, expand=1)
        
        # Initialize the figure attribute
        self.figure = Figure()  # Adjust figure size as needed

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.frame_id_label = tk.Label(self.left_frame, text="Frame ID: ")
        self.frame_id_label.pack(side='top')
        
        self.folder_path = os.path.join(os.getcwd(), 'matrices')
        self.csv_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.csv')])
        self.current_index = len(self.csv_files) - 1
        
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
            quantile_value = np.quantile(array, 0.98)
            array[array < quantile_value] = 0
            G = nx.DiGraph(array)
            if self.pos is None:
                self.pos = nx.spring_layout(G, k=0.4)
            pos = self.pos
            
            # Clear the figure before drawing the new frame
            self.figure.clf()
            
            fig = self.figure
            ax = fig.add_subplot(111)

            # Identify top 10 most influential nodes
            top_nodes = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:10]  # Changed from 5 to 10
            top_node_indices = [node for node, _ in top_nodes]

            # Prepare node colors
            node_colors = ['#FF6347' if i in top_node_indices else '#E0E0E0' for i in range(len(G.nodes))]

            # Prepare edge colors based on weights
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1  # Avoid division by zero
            edge_colors = [plt.cm.Blues(weight / max_weight) for weight in edge_weights]  # Use a colormap

            nx.draw(G, pos, ax=ax, with_labels=True, node_size=300, font_size=8, 
                    edge_color=edge_colors, arrows=True, labels=dict(enumerate(self.node_labels)), 
                    node_color=node_colors)
            
            # Adjust the margins
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            self.canvas.figure = fig
            self.canvas.draw()
            
            self.frame_id_label.config(text=f"Frame ID: {index + 1}")
            
            # Update the tables with top 10 most influential nodes
            self.update_tables(G)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load and process CSV file\n{e}")
    
    def update_tables(self, G, top_n=20):
        # Clear existing items
        for tree in [self.tree_out, self.tree_in]:
            for i in tree.get_children():
                tree.delete(i)
        
        # Get top N nodes with most outgoing edges, considering average weight
        out_degree_with_avg = [(node, degree, np.mean([G[node][succ]['weight'] for succ in G.successors(node)])) 
                               for node, degree in G.out_degree()]
        top_nodes_out = sorted(out_degree_with_avg, key=lambda x: (x[1], x[2]), reverse=True)[:top_n]
        
        for node, _, _ in top_nodes_out:
            node_label = self.node_labels[node]
            effects = sorted(G.successors(node), key=lambda x: G[node][x]['weight'], reverse=True)
            effect_labels = ', '.join([self.node_labels[e] for e in effects])
            self.tree_out.insert('', 'end', values=(node_label, effect_labels))
        
        # Get top N nodes with most incoming edges, considering average weight
        in_degree_with_avg = [(node, degree, np.mean([G[pred][node]['weight'] for pred in G.predecessors(node)])) 
                              for node, degree in G.in_degree()]
        top_nodes_in = sorted(in_degree_with_avg, key=lambda x: (x[1], x[2]), reverse=True)[:top_n]
        
        for node, _, _ in top_nodes_in:
            node_label = self.node_labels[node]
            causes = sorted(G.predecessors(node), key=lambda x: G[x][node]['weight'], reverse=True)
            cause_labels = ', '.join([self.node_labels[c] for c in causes])
            self.tree_in.insert('', 'end', values=(node_label, cause_labels))

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
    

