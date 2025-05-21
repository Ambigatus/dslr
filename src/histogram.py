#!/usr/bin/env python3
"""
histogram.py - A script to create histograms to analyze the score distribution
among Hogwarts houses for different courses.
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def load_data(file_path: str) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    """
    Load data from a CSV file and organize it by houses and features.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (feature_names, data_by_house) where data_by_house is organized by house and feature
    """
    feature_names = []
    data_by_house = {
        'Gryffindor': {},
        'Hufflepuff': {},
        'Ravenclaw': {},
        'Slytherin': {}
    }

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            # Find indices of relevant columns
            house_idx = header.index('Hogwarts House') if 'Hogwarts House' in header else None

            if house_idx is None:
                print("Error: 'Hogwarts House' column not found in the dataset.")
                sys.exit(1)

            # Get all feature names except Index and Hogwarts House
            feature_names = [name for i, name in enumerate(header) if i != 0 and name != 'Hogwarts House']

            # Initialize data structure
            for house in data_by_house:
                for feature in feature_names:
                    data_by_house[house][feature] = []

            # Read data values
            for row in reader:
                house = row[house_idx]
                if house in data_by_house:
                    for i, value in enumerate(row):
                        if i != 0 and i != house_idx and header[i] in feature_names:
                            feature = header[i]
                            try:
                                # Only add numerical values
                                if value:
                                    data_by_house[house][feature].append(float(value))
                            except ValueError:
                                # Skip non-numeric values
                                pass

        return feature_names, data_by_house
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def calculate_homogeneity(houses_data: Dict[str, List[float]]) -> float:
    """
    Calculate a homogeneity score for a feature across houses.
    Lower scores indicate more homogeneous distribution.

    Args:
        houses_data: Dictionary mapping house names to lists of feature values

    Returns:
        Homogeneity score (standard deviation of feature means across houses)
    """
    # Calculate mean for each house
    house_means = {}
    for house, values in houses_data.items():
        if values:
            house_means[house] = sum(values) / len(values)
        else:
            house_means[house] = 0

    # Calculate standard deviation of means
    mean_of_means = sum(house_means.values()) / len(house_means)
    variance = sum((m - mean_of_means) ** 2 for m in house_means.values()) / len(house_means)

    return variance ** 0.5


def find_most_homogeneous_feature(feature_names: List[str], data_by_house: Dict[str, Dict[str, List[float]]]) -> str:
    """
    Find the feature with the most homogeneous distribution across houses.

    Args:
        feature_names: List of feature names
        data_by_house: Data organized by house and feature

    Returns:
        Name of the most homogeneous feature
    """
    homogeneity_scores = {}

    for feature in feature_names:
        # Skip non-course features like First Name, Last Name, etc.
        if feature in ['First Name', 'Last Name', 'Birthday', 'Best Hand']:
            continue

        # Collect data for this feature from all houses
        houses_data = {house: data_by_house[house][feature] for house in data_by_house}

        # Calculate homogeneity score
        homogeneity_scores[feature] = calculate_homogeneity(houses_data)

    # Find feature with lowest homogeneity score (most homogeneous)
    return min(homogeneity_scores, key=homogeneity_scores.get)


def plot_histograms(feature: str, data_by_house: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Plot histograms for a feature across all houses.

    Args:
        feature: Feature name to plot
        data_by_house: Data organized by house and feature
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    plt.title(f'Distribution of {feature} Scores by Hogwarts House', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Set up colors and transparency
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'gold',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }
    alpha = 0.7

    # Create histograms
    for house, color in colors.items():
        data = data_by_house[house][feature]
        plt.hist(data, bins=15, alpha=alpha, color=color, label=house)

    # Add legend
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save figure
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f"{feature.replace(' ', '_')}_histogram.png"))

    # Show plot
    plt.tight_layout()
    plt.show()


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
    feature_names, data_by_house = load_data(file_path)

    # Find the most homogeneous feature
    most_homogeneous = find_most_homogeneous_feature(feature_names, data_by_house)

    print(f"The Hogwarts course with the most homogeneous score distribution between houses is: {most_homogeneous}")

    # Plot histograms for the most homogeneous feature
    plot_histograms(most_homogeneous, data_by_house)


if __name__ == "__main__":
    main()