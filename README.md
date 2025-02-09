# Time Series Causal Discovery

A Python-based tool for discovering and analyzing causal relationships in time series data, with a specific focus on financial market analysis. The project implements Linear Non-Gaussian Acyclic Models (LiNGAM) for causal discovery in temporal data.

## Features

- Time series causal discovery using VARLiNGAM (Vector Autoregressive LiNGAM)
- Interactive GUI for visualization and analysis
- Sliding window analysis for temporal causal relationships
- Support for large-scale financial datasets (e.g., S&P 100)
- Real-time causal graph generation and visualization
- Structural change detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/time-series-causal-discovery.git
cd time-series-causal-discovery
```

2. Install required dependencies:
```bash
pip install numpy networkx lingam matplotlib jupyter
```

## Project Structure

```
/
├── causal_graphs/      # Generated causal graph visualizations
├── predictions/        # Output predictions
├── backend/
│   └── fastlingam/    # Custom LiNGAM implementation
├── data/
│   └── sp100.csv      # Sample S&P 100 dataset
└── matrices/          # Generated correlation matrices
```

## Usage

### GUI Interface
To launch the interactive GUI:
```bash
python3 demo-gui.py
```

### Development Notebook
For detailed analysis and development:
1. Start Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `causal-discovery-dev.ipynb`

## Data Format

Input data should be in CSV format with:
- First row: Variable names/stock symbols
- Subsequent rows: Time series data
- No missing values

## Analysis Parameters

- Window Size: 400 time steps (configurable)
- VAR Lags: 2 (configurable)
- Step Size: 1 (for sliding window)

## Planned Improvements

* Add alerts when structural changes take place
* Implement additional causal discovery algorithms
* Enhanced visualization options
* Performance optimizations for large datasets
* Extended documentation and examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Contact

[Add your contact information here]
