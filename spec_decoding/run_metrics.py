#!/usr/bin/env python
"""
Metrics Runner for Speculative Decoding

This script runs benchmarks for different speculative decoding methods and generates
performance metrics including tokens per second, acceptance rates, and more.
"""

import torch
import argparse
import time
import json
import os
from typing import Dict, List
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Import from our speculative decoding script
from SPECULATIVE_DECODING import SpeculativeInferenceEngine

# Initialize colorama
init()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run metrics for Speculative Decoding methods')
    
    # Model configuration
    parser.add_argument('--draft_model', type=str, default="gpt2",
                        help='Name or path of the draft model')
    parser.add_argument('--application_model', type=str, default="gpt2-medium",
                        help='Name or path of the application model')
    parser.add_argument('--draft_quantization', type=str, default="none",
                        choices=["none", "int8", "int4", "fp8", "fp4"],
                        help='Quantization method for draft model')
    parser.add_argument('--application_quantization', type=str, default="none",
                        choices=["none", "int8", "int4", "fp8", "fp4"],
                        help='Quantization method for application model')
    
    # Benchmark parameters
    parser.add_argument('--prompt', type=str, 
                        default="Write a short story about a robot that learns to feel emotions.",
                        help='Prompt to use for benchmarking')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--gamma_values', type=str, default="2,4,6,8",
                        help='Comma-separated list of gamma values to test')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per configuration for averaging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default="metrics_results",
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    return parser.parse_args()

def run_benchmark(engine, args):
    """Run benchmarks for different methods and gamma values."""
    methods = ["autoregressive", "google", "deepmind", "qspec"]
    gamma_values = [int(g) for g in args.gamma_values.split(',')]
    
    # Create results directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    print(f"\n{Fore.CYAN}Running benchmarks with prompt:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{args.prompt}{Style.RESET_ALL}\n")
    
    # Run benchmarks
    for method in methods:
        print(f"\n{Fore.GREEN}Testing method: {method}{Style.RESET_ALL}")
        
        # For autoregressive, we only need to run once as gamma doesn't apply
        if method == "autoregressive":
            gamma_to_test = [gamma_values[0]]  # Just use the first gamma value as a placeholder
        else:
            gamma_to_test = gamma_values
        
        for gamma in gamma_to_test:
            print(f"  Testing with gamma = {gamma}")
            
            method_results = []
            
            # Run multiple times for averaging
            for run in range(args.runs):
                print(f"    Run {run+1}/{args.runs}...", end="", flush=True)
                
                # Set seed for reproducibility (different for each run)
                run_seed = args.seed + run
                
                start_time = time.time()
                result = engine.generate(
                    prompt=args.prompt,
                    method=method,
                    max_tokens=args.max_tokens,
                    gamma=gamma,
                    temperature=1.0,
                    top_k=20,
                    top_p=0.9,
                    seed=run_seed,
                    return_tokens=True
                )
                total_time = time.time() - start_time
                
                # Add run-specific data
                result["run"] = run + 1
                result["total_time"] = total_time
                
                method_results.append(result)
                
                print(f" done in {total_time:.2f}s ({result['tokens_per_second']:.2f} tokens/s)")
            
            # Calculate averages
            avg_tokens_per_second = sum(r["tokens_per_second"] for r in method_results) / len(method_results)
            avg_generation_time = sum(r["generation_time_seconds"] for r in method_results) / len(method_results)
            
            # Create summary
            summary = {
                "method": method,
                "gamma": gamma,
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_generation_time": avg_generation_time,
                "runs": method_results
            }
            
            results.append(summary)
            
            print(f"  {Fore.CYAN}Average: {avg_tokens_per_second:.2f} tokens/s, {avg_generation_time:.2f}s{Style.RESET_ALL}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            "config": vars(args),
            "results": results
        }, f, indent=2)
    
    print(f"\n{Fore.GREEN}Results saved to {results_file}{Style.RESET_ALL}")
    
    # Generate and display summary table
    display_results_table(results)
    
    # Generate plots
    generate_plots(results, args.output_dir, timestamp)
    
    return results, results_file

def display_results_table(results):
    """Display a summary table of the results."""
    table_data = []
    headers = ["Method", "Gamma", "Tokens/s", "Generation Time (s)"]
    
    for result in results:
        table_data.append([
            result["method"],
            result["gamma"],
            f"{result['avg_tokens_per_second']:.2f}",
            f"{result['avg_generation_time']:.2f}"
        ])
    
    print(f"\n{Fore.CYAN}Performance Summary:{Style.RESET_ALL}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def generate_plots(results, output_dir, timestamp):
    """Generate plots comparing the different methods."""
    # Extract data for plotting
    methods = sorted(set(r["method"] for r in results))
    gamma_values = sorted(set(r["gamma"] for r in results))
    
    # Prepare data for bar charts
    method_labels = []
    tokens_per_second = []
    generation_times = []
    
    for method in methods:
        for gamma in gamma_values:
            # Find matching result
            matching = [r for r in results if r["method"] == method and r["gamma"] == gamma]
            if matching:
                result = matching[0]
                method_labels.append(f"{method}\n(γ={gamma})")
                tokens_per_second.append(result["avg_tokens_per_second"])
                generation_times.append(result["avg_generation_time"])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot tokens per second
    bars1 = ax1.bar(method_labels, tokens_per_second, color='skyblue')
    ax1.set_title('Tokens per Second')
    ax1.set_ylabel('Tokens/s')
    ax1.set_ylim(bottom=0)
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Plot generation times
    bars2 = ax2.bar(method_labels, generation_times, color='salmon')
    ax2.set_title('Generation Time')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_ylim(bottom=0)
    
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"metrics_plot_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"{Fore.GREEN}Plot saved to {plot_file}{Style.RESET_ALL}")
    
    # Try to display the plot if running in an environment that supports it
    try:
        plt.show()
    except:
        pass

def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"{Fore.CYAN}Initializing Speculative Inference Engine...{Style.RESET_ALL}")
    engine = SpeculativeInferenceEngine(
        draft_model_name=args.draft_model,
        application_model_name=args.application_model,
        draft_quantization=args.draft_quantization,
        application_quantization=args.application_quantization,
        verbose=args.verbose
    )
    
    print(f"{Fore.CYAN}Starting benchmark...{Style.RESET_ALL}")
    results, results_file = run_benchmark(engine, args)
    
    print(f"\n{Fore.GREEN}Benchmark completed!{Style.RESET_ALL}")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
