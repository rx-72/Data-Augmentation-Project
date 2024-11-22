import numpy as np
import pandas as pd
from tqdm import tqdm
#UNTESTED, DONT KNOW IF IT IS EVEN CORRECT
# What does this do?: `bin_and_sample_feature` and `determine_optimal_bin_size`.

# `bin_and_sample_feature` function discretizes a specified feature of the training dataset into bins,
# with a given number of bins. Samples data points???
# ???? from these bins according to specified thresholds.
# ???The function returns the indices of the sampled data points and the edges of the bins used for discretization.

def bin_and_sample_feature(X_train, feature, thresholds, total_samples, num_bins):
    feature_values = X_train[feature].clip(lower=X_train[feature].min(), upper=X_train[feature].max())
    bin_edges = np.linspace(feature_values.min(), feature_values.max(), num_bins + 1)
    bin_counts, _ = pd.cut(feature_values, bins=bin_edges, labels=False, retbins=True)

    selected_bins_by_threshold = []
    for thresh in thresholds:
        selected_bins = [bin_idx for bin_idx in range(num_bins)
                         if thresh[0] <= bin_edges[bin_idx] and thresh[1] >= bin_edges[bin_idx + 1]]
        selected_bins_by_threshold.append(selected_bins)

    sampled_indices = set()
    for selected_bins in selected_bins_by_threshold:
        bin_freqs = feature_values.groupby(bin_counts).size().reindex(range(num_bins), fill_value=0)
        bin_priority = sorted(selected_bins, key=lambda bin_idx: -bin_freqs[bin_idx])

        for bin_idx in bin_priority:
            bin_indices = feature_values[bin_counts == bin_idx].index
            sampled_indices.update(bin_indices[:total_samples - len(sampled_indices)])
            if len(sampled_indices) >= total_samples:
                break
        if len(sampled_indices) >= total_samples:
            break

    return list(sampled_indices), bin_edges

#FORGOT WHAT I WAS DOING HERE, DON'T know what im returning
# `determine_optimal_bin_size` function evaluates different bin sizes to find the optimal bin count??
# LOST TRACK HERE?
# that minimizes the robustness ratio for a given feature. It iterates over a range of bin sizes, calculates
# the robustness ratio for each bin size using the `bin_and_sample_feature` function, and identifies the bin
# count that results in the lowest robustness ratio. The function returns the optimal bin count and the 
# corresponding bin edges and robustness ratio.

def determine_optimal_bin_size(X_train, y_train, X_test, y_test, feature, thresholds, ratio, robustness_radius, uncertain_percentage):
    label_range = y_train.max() - y_train.min()
    uncertain_radius = ratio * label_range
    min_robustness_ratio = float('inf')
    optimal_bin_count = None
    best_result = None

    for num_bins in tqdm(range(5, 51), desc='Testing bin sizes'):
        sampled_indices, bin_edges = bin_and_sample_feature(
            X_train, feature, thresholds,
            total_samples=int(uncertain_percentage * len(X_train)),
            num_bins=num_bins
        )
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(
            X_train, y_train, X_test, y_test,
            uncertain_num=int(uncertain_percentage * len(X_train)),
            boundary_indices=np.array(sampled_indices),
            uncertain_radius