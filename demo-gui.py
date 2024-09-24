import tkinter as tk
import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox

class CSVViewer:
    def __init__(self, master):
        self.pos = None
        self.master = master
        self.master.title("Causal relationships in S&P 100 (500-day window)")
        
        # Initialize the figure attribute
        self.figure = Figure(figsize=(14, 8))

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.frame = tk.Frame(self.master)  # Ensure frame is initialized
        self.frame.pack()
        
        self.frame_id_label = tk.Label(self.master, text="Frame ID: ")
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
        self.shortcut_label = tk.Label(self.master, text="Shortcuts: Left Arrow - Previous Frame, Right Arrow - Next Frame", bg='white', fg='black')
        self.shortcut_label.place(relx=0.5, rely=0.95, anchor='center')

        
    def load_frame(self, index):
        file_path = os.path.join(self.folder_path, self.csv_files[index])
        
        try:
            array = pd.read_csv(file_path, header=None).values.T
            quantile_value = np.quantile(array, 0.99)
            array[array < quantile_value] = 0
            G = nx.DiGraph(array)
            if self.pos is None:
                self.pos = nx.spring_layout(G, k=0.5)
            pos = self.pos
            
            # Clear the figure before drawing the new frame
            self.figure.clf()
            
            fig = self.figure
            ax = fig.add_subplot(111)
            nx.draw(G, pos, ax=ax, linewidths= 1.0, edgecolors = 'black', node_color = 'none', node_size=860, font_size=8, edge_color='blue', arrows=True, with_labels=True, labels=dict(enumerate(self.node_labels)))
            
            # Adjust the margins
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            self.canvas.figure = fig
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(expand=1, fill='both')
            
            self.frame_id_label.config(text=f"Frame ID: {index + 1}")
            
            # Resize the figure to match the window size
            self.master.update_idletasks()
            width = self.master.winfo_width()
            height = self.master.winfo_height()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load and process CSV file\n{e}")
    
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
    app = CSVViewer(root)
    root.mainloop()

