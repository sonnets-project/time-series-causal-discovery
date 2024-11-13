import argparse
import numpy as np
from common import granger_causality_matrix  # Import the function
import os

def process_csv(input_file, output_file=None, max_lags=5, significance_level=0.05, metric='is_causal'):
    # Read data
    raw_data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    
    # Compute Granger causality matrix
    causality_matrix, detailed_results = granger_causality_matrix(
        data=raw_data,
        max_lags=max_lags,
        significance_level=significance_level,
        metric=metric
    )
    
    # Output the causality matrix
    output = "Granger Causality Matrix:\n" + str(causality_matrix)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
    else:
        print(output)

def main():
    parser = argparse.ArgumentParser(description='Discover Granger causality relationships in time series data.')
    parser.add_argument('-i', '--input', help='Input CSV file path (default: standard input)', default=None)
    parser.add_argument('-o', '--output', help='Output file path (default: standard output)', default=None)
    parser.add_argument('--max_lags', type=int, default=5, help='Maximum number of lags to test (default: 5)')
    parser.add_argument('--significance_level', type=float, default=0.05, help='Significance level for the test (default: 0.05)')
    parser.add_argument('--metric', type=str, default='is_causal', choices=['is_causal', 'min_pvalue', 'max_fstat'],
                        help='Metric to use for the result matrix (default: is_causal)')
    
    args = parser.parse_args()
    
    input_file = args.input if args.input else 'stdin'  # Use 'stdin' if no input file is provided
    output_file = args.output  # Use provided output file or None for standard output
    process_csv(input_file, 
                 output_file=output_file, 
                 max_lags=args.max_lags, 
                 significance_level=args.significance_level, 
                 metric=args.metric)

if __name__ == "__main__":
    main()
