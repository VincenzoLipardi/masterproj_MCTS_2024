import os

import statistics
import pandas as pd
import numpy as np
from qiskit import qpy
import matplotlib
import matplotlib.pyplot as plt
import json
from itertools import cycle
import math

import evaluation_functions as evf


def pre_post_plot(problems:list, 
                  predefined_color:dict, 
                  variants:list = None,
                  optimal_value:dict = {'h2': 1.117, 'h2o': 75.4217, 'lih': 7.973, 'vqls_0': 0, 'vqls_1': 0}, 
                  folder_dir: str = "results", 
                  dirname: str = os.getcwd(),
                  dot_gradient: str = None):
    
    test_results_path = os.path.join(dirname, folder_dir)
    csvs = {}

    for variant_name in os.listdir(test_results_path):
        variant_path = os.path.join(test_results_path, variant_name)

        for problem_name in os.listdir(variant_path):
            folder_path = os.path.join(variant_path, problem_name)
            pre_values = []
            post_values = []

            if os.path.isdir(folder_path):  # Check if it's a directory
                if problem_name not in csvs:
                    csvs[problem_name] = {}

                if variant_name not in csvs[problem_name]:
                    csvs[problem_name][variant_name] = {}

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    if os.path.isfile(file_path) and file_name.endswith(".csv") and "optimized" in file_name:
                        df = pd.read_csv(file_path)
                        df.set_index('Category', inplace=True)
                        pre_values.append(max([df.loc["last", "ValueBefore"], df.loc["best", "ValueBefore"]]))
                        post_values.append(max([df.loc["last", "ValueAfter"], df.loc["best", "ValueAfter"]]))
                csvs[problem_name][variant_name]["pre"] = np.average(pre_values)
                csvs[problem_name][variant_name]["post"] = np.average(post_values)

    for problem in problems:
        # Get the variants for the chosen problem
        
        variants_dict = csvs[problem]
        if variants:
            variant_names = variants
        else:
            variant_names = list(variants_dict.keys())
        print(variant_names)
        pre_values = [variants_dict[variant]['pre'] for variant in variant_names]
        post_values = [variants_dict[variant]['post'] for variant in variant_names]

        # Plot
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for i, variant in enumerate(variant_names):
            color = predefined_color[variant]
            plt.plot([pre_values[i], post_values[i]], [i, i], color=color, linestyle='-', alpha=1)
            if dot_gradient == "hollow":
                plt.plot(pre_values[i], i, marker='4', color=color, label=variant)
                plt.plot(post_values[i], i, marker='3', color=color, label=variant)
            elif dot_gradient == "alpha":
                plt.plot(pre_values[i], i, marker='o', color=color, alpha=0.5, label=variant)
                plt.plot(post_values[i], i, marker='o', color=color, alpha=1, label=variant)
            else:
                plt.plot([pre_values[i], post_values[i]], [i, i], marker='o', color=color, label=variant)
        ax.axvline(x=optimal_value[problem], color='r', linestyle='--', label='Optimal Value')

        # Add labels and title
        plt.yticks(range(len(variant_names)), variant_names)
        plt.xlabel('Values')
        plt.ylabel('Variants')
        plt.title(f'Pre vs. Post Values for {problem}')
        plt.grid(linestyle='--', alpha=0.6)
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Variants")
        plt.tight_layout()
        plt.show()


def load_mcts_results(folder_dir: str = "results", dirname: str = os.getcwd()):
    """
    Load all the csv files in the results folder

    returns:
    {
        "problem": {
            "variant": [
                pd.DataFrame,
                pd.DataFrame,
                ...
            ],
            ...s
    """
    test_results_path = os.path.join(dirname, folder_dir)
    csvs = {}

    for variant_name in os.listdir(test_results_path):
        variant_path = os.path.join(test_results_path, variant_name)

        for problem_name in os.listdir(variant_path):
            folder_path = os.path.join(variant_path, problem_name)

            if os.path.isdir(folder_path):  # Check if it's a directory
                if problem_name not in csvs:
                    csvs[problem_name] = {}

                if variant_name not in csvs[problem_name]:
                    csvs[problem_name][variant_name] = []

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    if os.path.isfile(file_path) and file_name.endswith(".csv") and not "optimized" in file_name:
                        csvs[problem_name][variant_name].append(pd.read_csv(file_path))
    return csvs


def load_circuit_results(
    folder_dir: str = "results", dirname: str = os.getcwd(), runs: int = 10
):
    """
    Load all the circuits in the results folder

    :param gradient_decent: If True, it will use the evaluation function with gradient

    returns:
    {
        "problem": {
            "variant": [val, val, val, ...],
            ...
    """
    test_results_path = os.path.join(dirname, folder_dir)
    circuits = {}

    problems = {
        "h2": evf.h2,
        "h2o": evf.h2o,
        "lih": evf.lih,
        "vqls_0": evf.vqls_0, # TODO: this does not have a gradient
        "vqls_1": evf.vqls_1,
    }

    for variant_name in os.listdir(test_results_path):
        variant_path = os.path.join(test_results_path, variant_name)

        for problem_name in os.listdir(variant_path):
            folder_path = os.path.join(variant_path, problem_name)

            if os.path.isdir(folder_path):  # Check if it's a directory
                if problem_name not in circuits:
                    circuits[problem_name] = {}

                if variant_name not in circuits[problem_name]:
                    circuits[problem_name][variant_name] = []
                                    
                for run in range(runs):
                    for file_name in os.listdir(folder_path)[::-1]:
                        file_path = os.path.join(folder_path, file_name)

                        if os.path.isfile(file_path) and file_name.endswith(
                            f"{run}_optimized.csv"
                        ):
                            with open(file_path, "rb") as handle:
                                df = pd.read_csv(handle)
                                max_val = df['ValueAfter'].max()
                                min_val = df['ValueAfter'].min()
                                if "vqls_0" in problem_name:
                                    circuits[problem_name][variant_name].append(np.exp((-min_val-1)*10))
                                elif "vqls_1" in problem_name:
                                    circuits[problem_name][variant_name].append(np.exp(max_val * 10))
                                else:
                                    circuits[problem_name][variant_name].append(max_val)

    return circuits


def load_circuit_gates(
    folder_dir: str = "results", dirname: str = os.getcwd(), runs: int = 10
):
    """
    Load all the circuit gate counts in the results folder

    returns:
    {
        "problem": {
            "variant": [val, val, val, ...],
            ...
    """
    test_results_path = os.path.join(dirname, folder_dir)
    circuits = {}

    problems = {
        "h2": evf.h2,
        "h2o": evf.h2o,
        "lih": evf.lih,
        "vqls_0": evf.vqls_0, # TODO: this does not have a gradient
        "vqls_1": evf.vqls_1,
    }

    for variant_name in os.listdir(test_results_path):
        variant_path = os.path.join(test_results_path, variant_name)

        for problem_name in os.listdir(variant_path):
            folder_path = os.path.join(variant_path, problem_name)

            if os.path.isdir(folder_path):  # Check if it's a directory
                if problem_name not in circuits:
                    circuits[problem_name] = {}

                if variant_name not in circuits[problem_name]:
                    circuits[problem_name][variant_name] = {}
                                    
                best_value = float("-inf")
                for run in range(runs):
                    for file_name in os.listdir(folder_path)[::-1]:
                        file_path = os.path.join(folder_path, file_name)

                        if os.path.isfile(file_path) and file_name.endswith(
                            f"{run}_optimized.csv"
                        ):
                            with open(file_path, "rb") as handle:
                                df = pd.read_csv(handle)
                                id = df['ValueAfter'].idxmin() if "vqls_0" in problem_name else df['ValueAfter'].idxmax()
                                if (best_value < df.loc[id, 'ValueAfter'] and "vqls_0" not in problem_name) or (best_value > df.loc[id, 'ValueAfter'] and "vqls_0" in problem_name):
                                    best_value = df.loc[id, 'ValueAfter']
                                    rx = df.loc[id, 'Rx']
                                    ry = df.loc[id, 'Ry']
                                    rz = df.loc[id, 'Rz']
                                    
                                    circuits[problem_name][variant_name] = {
                                        "h": df.loc[id, 'H'],
                                        "r": rx + ry + rz,
                                        "rx": rx,
                                        "ry": ry,
                                        "rz": rz,
                                        "cx": df.loc[id, 'Cx'],
                                    }
    return circuits
    

def converge_all(results: dict, problems: list, variants: list, budget: int = 5000):
    converged = {}
    for problem in problems:
        variant_results = []
        for variant in variants:
            conv = get_mcts_convergence(results[problem][variant], budget, problem)
            variant_results.append(
                {
                    "variant": variant,
                    "convergence": conv['averages'],
                    "std_devs": conv['std_devs'],
                }
            )
        converged[problem] = variant_results
    return converged

def plot_figure(problems: str,  
                variants: list, 
                optimal_value: float,
                convergence: dict,
                predefined_colors: dict,
                baseline: str = "standard_nc",
                std_dev: str = "shaded",
                alpha: float = 0.05,
                figsize: tuple = (12, 8), 
                n_cols: int = 3,
                errorbar_interval: int = 100):

    """
    Plots convergence for a list of problems in a grid layout, with consistent variant colors and a single legend.

    Parameters:
    - problems: List of problem names to plot.
    - results: Dictionary of results (not used here, but passed for compatibility).
    - variants: List of variant names.
    - optimal_value: Dictionary mapping problem names to optimal values.
    - convergence: Dictionary mapping problem names to their convergence data.
    - std_dev: String indicating how to display standard deviation ("shaded" or "dotted").
    - figsize: Tuple for the figure size.
    - n_cols: Number of columns in the grid layout.
    """
    # Predefine colors for variants
    
    # Determine number of rows and columns for the grid
    n_rows = math.ceil(len(problems) / n_cols)

    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot each problem
    for i, problem in enumerate(problems):
        ax = axes[i]
        variant_results = convergence[problem]

        target_variants = []
        for c in predefined_colors.keys():
            if predefined_colors[c] != "gainsboro":
                target_variants.append(c)

        sorted_variants = sorted(
        variant_results, 
        key=lambda x: x["variant"] in target_variants
        )
        variant_results = sorted_variants

        # Plot each variant's convergence
        for entry in variant_results:
            name = entry["variant"]
            if name not in variants:  # Skip if the variant is not in the 'variants' list
                continue
            values = entry["convergence"]
            std_devs = entry["std_devs"]
            x_range = range(len(values))

            # Get the color for the current variant
            color = predefined_colors[name]

            # Plot the variant's line
            ax.plot(x_range, values, color=color, marker='', linestyle='-', label=name)

            # Plot standard deviation as shaded or dotted lines
            if std_dev == "shaded" and predefined_colors[name] != "gainsboro":
                ax.fill_between(
                    x_range,
                    [v - s for v, s in zip(values, std_devs)],  # Lower bound
                    [v + s for v, s in zip(values, std_devs)],  # Upper bound
                    color=color,
                    alpha=alpha
                )
            elif std_dev == "dotted" and predefined_colors[name] != "gainsboro":
                lower_bound = [v - s for v, s in zip(values, std_devs)]
                upper_bound = [v + s for v, s in zip(values, std_devs)]
                ax.plot(x_range, lower_bound, linestyle=':', color=color, alpha=alpha, label="_nolegend_")
                ax.plot(x_range, upper_bound, linestyle=':', color=color, alpha=alpha, label="_nolegend_")
            elif std_dev == "errorbar" and predefined_colors[name] != "gainsboro":
                ax.errorbar(x_range[::errorbar_interval], values[::errorbar_interval], yerr=std_devs[::errorbar_interval], fmt='o', color=color, alpha=alpha, label="_nolegend_")

        # Add optimal value line
        ax.axhline(y=optimal_value[problem], color='r', linestyle='--', label='Optimal Value')
        ax.set_xlabel('Budget', fontsize=12)
        ax.set_ylabel('Cost' if 'vqls' in problem else 'Reward', fontsize=12)

        # Add title and grid
        ax.set_title(f'QAS problem: {problem}', fontsize=16)
        ax.grid(True)

    # Hide unused subplots
    for ax in axes[len(problems):]:
        ax.set_visible(False)

    # Extract the handles and labels from the first subplot (for the legend)
    handles, labels = axes[0].get_legend_handles_labels()

    # Move the baseline (if present) to the beginning of the legend
    baseline_handle = None
    baseline_label = None
    filtered_handles = []
    filtered_labels = []

    # Filter out gainsboro labels and identify the baseline
    for handle, label in zip(handles, labels):
        if label == baseline:
            baseline_handle = handle
            baseline_label = label
        elif label != "Optimal Value" and predefined_colors[label] != "gainsboro":
            filtered_handles.append(handle)
            filtered_labels.append(label)

    # If the baseline exists, reorder the handles and labels
    if baseline_handle is not None:
        handles = [baseline_handle] + filtered_handles
        labels = [baseline_label] + filtered_labels
    else:
        handles = filtered_handles
        labels = filtered_labels


    # Add a single legend in the bottom-right panel (6th panel for 3 columns)
    legend_ax = fig.add_subplot(n_rows, n_cols, len(axes))  # Create an axes in the last position
    legend_ax.axis('off')  # Hide the axes
    # handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center', ncol=1, fontsize=14)  # Adjust legend settings


    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for legend at the bottom
    plt.show()


def get_mcts_convergence(
    dataframes: list = [],
    max_budget: int = None,
    problem: str = "h2",
):
    """
    Gets best objective value found for each budget for the given problem + variant

    returns: list with the objective value for each budget increment
    """
    all_tests = []
    if not max_budget:
        max_budget = min(df["budget"].max() for df in dataframes)

    for dataframe in dataframes:
        best_per_budget = []  # contains the best value for this tests
        for budget in range(1, max_budget):
            if problem == "vqls_0" or problem == "vqls_1":
                max_value = np.exp(-10 * dataframe[dataframe["budget"] <= budget][
                    "objectiveValue"
                ].max())
            else:
                max_value = dataframe[dataframe["budget"] <= budget][
                    "objectiveValue"
                ].max()
            best_per_budget.append(max_value)
        all_tests.append(best_per_budget)

    transposed = list(zip(*all_tests))

    # Calculate the statistics
    averages = [sum(values) / len(values) for values in transposed]
    std_devs = [statistics.stdev(values) if len(values) > 1 else 0 for values in transposed]
    variances = [statistics.variance(values) if len(values) > 1 else 0 for values in transposed]

    return {
        "averages": averages,
        "std_devs": std_devs,
        "variances": variances,
    }

    # return [
    #     sum(values) / len(values) for values in zip(*all_tests)
    # ]  # averages over all sims

def make_box_plot(data: dict, optimal_value: float):

    results = {}
    for problem in data.keys(): 
        results[problem] = {}
        for variant in data[problem].keys():
            best_values = []
            for run in data[problem][variant]:
                if run:
                    value = float('inf')
                    if run['best']:
                        value = min(run['best']['value'][-1])
                    if run['greedy']:
                        value = min(value, run['greedy']['value'][-1])
                    if problem != 'vqls_0' and problem != 'vqls_1':
                        best_values.append(value * -1)
                    else:
                        best_values.append(value)

            avg = np.mean(best_values)
            std_dev = np.std(best_values)
            var = np.var(best_values)

            # Store statistics
            results[problem][variant] = {
                "values": best_values,
                "average": avg,
                "std_dev": std_dev,
                "variance": var,
            }

    # Plot the data
    for problem, variants in results.items():
        plt.figure(figsize=(10, 6))
        plt.title(f"Box Plot for {problem}")
        plt.xlabel("Variants")
        plt.ylabel("Values")

        boxplot_data = [variants[variant]["values"] for variant in variants.keys()]
        variant_names = list(variants.keys())

        plt.boxplot(boxplot_data, labels=variant_names, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.axhline(y=optimal_value[problem], color="red", linestyle="--", label="Optimal Value")
        plt.grid()
        plt.tight_layout()

def load_json(path: str = "results.json"):
    file_path = os.path.join(os.getcwd(), path)

    # JSON
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
def standardize_problem_stats(problem_values: list):
    standardized_results = []
    
    scores = np.array(problem_values)
    # Z-score standardization
    mean = np.mean(scores)
    std = np.std(scores)
    standardized = (scores - mean) / (std if std != 0 else 1)  # Avoid division by zero
    standardized_results.append(standardized.tolist())
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    stats = {   "mean": mean,
                "std": std,
                "min": min_score,
                "max": max_score
            }
    return stats

def standardize_from_stats(stats, score):
    z_score = (score - stats["mean"]) / (stats["std"] if stats["std"] != 0 else 1)
    min_max_normalized = (score - stats["min"]) / (stats["max"] - stats["min"]) if stats["max"] != stats["min"] else 0
    return z_score

if __name__ == "__main__":
    results = load_circuit_results()
    print(results['vqls_0'])
   
