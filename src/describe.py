#!/usr/bin/env python3
"""
describe.py - A script to analyze and display statistics for numerical features in a dataset.
This is part of the Hogwarts Houses Sorting project.
"""

import os
import sys
import csv
from typing import List, Dict, Tuple, Optional


def load_data(file_path: str) -> Tuple[List[str], List[List[Optional[float]]]]:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (feature_names, data) where data is a list of lists of float values
    """
    feature_names = []
    data = []

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            feature_names = header[1:]  # Skip the Index column

            # Initialize data structure
            for _ in range(len(feature_names)):
                data.append([])

            # Read data values
            for row in reader:
                for i in range(1, len(row)):
                    try:
                        value = float(row[i]) if row[i] else None
                        data[i - 1].append(value)
                    except ValueError:
                        # Skip non-numeric values
                        data[i - 1].append(None)

        return feature_names, data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def count_non_null(values: List[Optional[float]]) -> int:
    """
    Count non-null values in a list.

    Args:
        values: List of values

    Returns:
        Count of non-null values
    """
    return sum(1 for v in values if v is not None)


def calculate_mean(values: List[Optional[float]]) -> float:
    """
    Calculate mean of non-null values.

    Args:
        values: List of values

    Returns:
        Mean value
    """
    non_null_values = [v for v in values if v is not None]
    return sum(non_null_values) / len(non_null_values) if non_null_values else float('nan')


def calculate_std(values: List[Optional[float]]) -> float:
    """
    Calculate standard deviation of non-null values.

    Args:
        values: List of values

    Returns:
        Standard deviation
    """
    non_null_values = [v for v in values if v is not None]

    if not non_null_values or len(non_null_values) < 2:
        return float('nan')

    mean = calculate_mean(non_null_values)
    variance = sum((x - mean) ** 2 for x in non_null_values) / len(non_null_values)
    return variance ** 0.5


def calculate_min(values: List[Optional[float]]) -> float:
    """
    Find minimum value.

    Args:
        values: List of values

    Returns:
        Minimum value
    """
    non_null_values = [v for v in values if v is not None]
    return min(non_null_values) if non_null_values else float('nan')


def calculate_max(values: List[Optional[float]]) -> float:
    """
    Find maximum value.

    Args:
        values: List of values

    Returns:
        Maximum value
    """
    non_null_values = [v for v in values if v is not None]
    return max(non_null_values) if non_null_values else float('nan')


def calculate_percentile(values: List[Optional[float]], percentile: float) -> float:
    """
    Calculate percentile value.

    Args:
        values: List of values
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value
    """
    non_null_values = [v for v in values if v is not None]

    if not non_null_values:
        return float('nan')

    # Sort values
    sorted_values = sorted(non_null_values)
    n = len(sorted_values)

    # Calculate index
    index = (n - 1) * (percentile / 100)

    # If index is an integer, return the value at that index
    if index.is_integer():
        return sorted_values[int(index)]

    # Otherwise, interpolate between the two nearest values
    lower_index = int(index)
    upper_index = min(lower_index + 1, n - 1)  # Ensure we don't go out of bounds

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    fraction = index - lower_index

    return lower_value + fraction * (upper_value - lower_value)


def is_numeric_feature(values: List[Optional[float]]) -> bool:
    """
    Determine if a feature is numeric.

    Args:
        values: List of values for a feature

    Returns:
        True if the feature has at least one non-null numeric value, False otherwise
    """
    # If more than 90% of values are None or we have less than 5 non-null values,
    # it's likely not a numeric feature
    non_null_values = [v for v in values if v is not None]
    return len(non_null_values) > 5 and len(non_null_values) / len(values) > 0.1


def calculate_statistics(data: List[List[Optional[float]]]) -> List[Dict]:
    """
    Calculate statistics for each feature.

    Args:
        data: List of lists of values

    Returns:
        List of dictionaries with statistics for each feature
    """
    stats = []

    for feature_values in data:
        if is_numeric_feature(feature_values):
            feature_stats = {
                'Count': count_non_null(feature_values),
                'Mean': calculate_mean(feature_values),
                'Std': calculate_std(feature_values),
                'Min': calculate_min(feature_values),
                '25%': calculate_percentile(feature_values, 25),
                '50%': calculate_percentile(feature_values, 50),
                '75%': calculate_percentile(feature_values, 75),
                'Max': calculate_max(feature_values)
            }
            stats.append(feature_stats)
        else:
            # For non-numeric features, create a stats dict with only count
            stats.append({
                'Count': count_non_null(feature_values),
                'Mean': float('nan'),
                'Std': float('nan'),
                'Min': float('nan'),
                '25%': float('nan'),
                '50%': float('nan'),
                '75%': float('nan'),
                'Max': float('nan')
            })

    return stats


def display_statistics(feature_names: List[str], stats: List[Dict]) -> None:
    """
    Display statistics in a formatted table.

    Args:
        feature_names: Names of features
        stats: Statistics for each feature
    """
    # Define column widths
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    feature_width = max(len(name) for name in feature_names) + 2
    stat_width = 15

    # Print header
    print(" " * feature_width, end="")
    for stat_name in stat_names:
        print(f"{stat_name:>{stat_width}}", end="")
    print()

    # Print statistics for each feature
    for i, feature_name in enumerate(feature_names):
        # Skip features with no numeric data
        if all(isinstance(stats[i][stat_name], float) and
               (stats[i][stat_name] != stats[i][stat_name]) for stat_name in stat_names if stat_name != 'Count'):
            continue

        print(f"{feature_name:<{feature_width}}", end="")

        for stat_name in stat_names:
            value = stats[i][stat_name]

            if stat_name == 'Count':
                print(f"{value:>{stat_width}d}", end="")
            elif isinstance(value, float) and value != value:  # Check for NaN
                print(f"{'NaN':>{stat_width}}", end="")
            else:
                print(f"{value:>{stat_width}.6f}", end="")

        print()


def main():
    """Main function to execute the script."""
    # Default file path - can be changed or read from arguments
    default_file_path = "data/dataset_train.csv"

    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file_path

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    # Load data
    feature_names, data = load_data(file_path)

    # Calculate statistics
    stats = calculate_statistics(data)

    # Display statistics
    display_statistics(feature_names, stats)


if __name__ == "__main__":
    main()