#!/usr/bin/env python3
"""
pair_plot.py - A script to create a scatter plot matrix (pair plot)
to visualize patterns that distinguish students from different Hogwarts houses.
This visualization helps identify features for logistic regression.
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

def remove_outliers(data: List[Optional[float]], z_thresh=3) -> List[Optional[float]]:
    clean = [v for v in data if v is not None]
    if len(clean) < 3:
        return data
    mean = np.mean(clean)
    std = np.std(clean)
    return [v if v is None or abs((v - mean) / std) <= z_thresh else None for v in data]

def normalize_data(data: List[Optional[float]]) -> List[Optional[float]]:
    clean = [v for v in data if v is not None]
    if not clean:
        return data
    min_v, max_v = min(clean), max(clean)
    if min_v == max_v:
        return [0 if v is not None else None for v in data]
    return [(v - min_v) / (max_v - min_v) if v is not None else None for v in data]

def load_data(file_path: str, normalize=False, filter_outliers=False) -> Tuple[List[str], Dict[str, Dict[str, List[Optional[float]]]]]:
    feature_names = []
    data_by_house = {}

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            house_idx = header.index('Hogwarts House')

            potential_features = [name for i, name in enumerate(header) if i != 0 and name != 'Hogwarts House']
            rows = list(reader)

            numeric_features = []
            for feature in potential_features:
                col_idx = header.index(feature)
                numeric_values = 0
                total_values = 0
                for row in rows:
                    if row[col_idx]:
                        total_values += 1
                        try:
                            float(row[col_idx])
                            numeric_values += 1
                        except ValueError:
                            pass
                if total_values > 0 and numeric_values / total_values >= 0.9:
                    numeric_features.append(feature)

            feature_names = numeric_features
            for row in rows:
                house = row[house_idx]
                if house not in data_by_house:
                    data_by_house[house] = {feature: [] for feature in feature_names}
                for feature in feature_names:
                    col_idx = header.index(feature)
                    try:
                        value = float(row[col_idx]) if row[col_idx] else None
                        data_by_house[house][feature].append(value)
                    except ValueError:
                        data_by_house[house][feature].append(None)

            # Apply filtering and normalization
            for house_data in data_by_house.values():
                for feature in feature_names:
                    if filter_outliers:
                        house_data[feature] = remove_outliers(house_data[feature])
                if normalize:
                    for feature in feature_names:
                        house_data[feature] = normalize_data(house_data[feature])

        return feature_names, data_by_house
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def calculate_feature_importance(feature_name: str, data_by_house: Dict[str, Dict[str, List[float]]]) -> float:
    house_stats = {}
    for house, features in data_by_house.items():
        values = [v for v in features[feature_name] if v is not None]
        if values:
            mean = sum(values) / len(values)
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            house_stats[house] = (mean, std)

    houses = list(house_stats.keys())
    between_variance = 0
    valid_pairs = 0
    for i in range(len(houses)):
        for j in range(i + 1, len(houses)):
            mean1, std1 = house_stats[houses[i]]
            mean2, std2 = house_stats[houses[j]]
            if std1 > 1e-6 and std2 > 1e-6:
                between_variance += abs(mean1 - mean2) / ((std1 + std2) / 2)
                valid_pairs += 1

    return between_variance / valid_pairs if valid_pairs > 0 else 0

def select_best_features(feature_names: List[str], data_by_house: Dict[str, Dict[str, List[float]]], num_features: int, verbose: bool = False) -> List[str]:
    feature_scores = [(f, calculate_feature_importance(f, data_by_house)) for f in feature_names]
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print("Feature importance scores:")
        for f, score in feature_scores:
            print(f"{f}: {score:.4f}")

    return [f for f, _ in feature_scores[:num_features]]

def create_pair_plot(data_by_house: Dict[str, Dict[str, List[Optional[float]]]], selected_features: List[str]) -> None:
    num_features = len(selected_features)
    house_colors = {'Gryffindor': 'red', 'Hufflepuff': 'gold', 'Ravenclaw': 'blue', 'Slytherin': 'green'}

    fig, axes = plt.subplots(num_features, num_features, figsize=(2.5 * num_features, 2.5 * num_features),
                             sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            x_feat, y_feat = selected_features[j], selected_features[i]
            if i == j:
                for house, house_data in data_by_house.items():
                    vals = [v for v in house_data[x_feat] if v is not None]
                    ax.hist(vals, bins=15, alpha=0.5, color=house_colors.get(house, 'gray'), label=house)
                ax.set_title(x_feat, fontsize=10)
            else:
                for house, house_data in data_by_house.items():
                    x = house_data[x_feat]
                    y = house_data[y_feat]
                    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
                    if pairs:
                        x_data, y_data = zip(*pairs)
                        ax.scatter(x_data, y_data, s=10, alpha=0.5, color=house_colors.get(house, 'gray'))
            if i == num_features - 1:
                ax.set_xlabel(x_feat, fontsize=8)
            if j == 0:
                ax.set_ylabel(y_feat, fontsize=8)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=6) for c in house_colors.values()]
    labels = list(house_colors.keys())
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    fig.suptitle('Pair Plot of Selected Features by House', fontsize=14)

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/pair_plot.png', dpi=300)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Create a pair plot for Hogwarts houses.")
    parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="Path to dataset CSV")
    parser.add_argument("--features", type=int, default=4, choices=range(2, 6), metavar="[2-5]", help="Number of top features to select (2-5)")
    parser.add_argument("--verbose", action="store_true", help="Print additional information")
    parser.add_argument("--normalize", action="store_true", help="Apply Min-Max normalization")
    parser.add_argument("--filter-outliers", action="store_true", help="Remove outliers with Z-score method")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    print(f"Loading data from {args.file}...")
    feature_names, data_by_house = load_data(args.file, normalize=args.normalize, filter_outliers=args.filter_outliers)

    if not feature_names:
        print("No valid numeric features found.")
        sys.exit(1)

    if args.features > len(feature_names):
        print(f"Only {len(feature_names)} numeric features available. Using all of them.")
        args.features = len(feature_names)

    selected = select_best_features(feature_names, data_by_house, args.features, verbose=args.verbose)
    create_pair_plot(data_by_house, selected)
    print("\nPair plot saved to outputs/pair_plot.png")

if __name__ == "__main__":
    main()
