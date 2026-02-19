from math import isclose
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
from deterministic_iterative_stratification import deterministic_iterative_train_test_split

from sklearn.model_selection import KFold

def generate_crossval_splits(case_ids: List[str], n_folds: int, seed: int) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for _, (train_indices, val_indices) in enumerate(kfold.split(case_ids)):
        train_keys = np.array(case_ids)[train_indices]
        val_keys = np.array(case_ids)[val_indices]
        splits.append({"train": list(train_keys), "val": list(val_keys)})
    return splits


def stratified_sampling(
    labels: np.ndarray,
    n_splits: Literal[2, 3],
    proportion_per_split: List[float],
    seed: int,
):
    """
    Args:
        labels (np.ndarray):
            - For multi-label task: 0-1-valued array of shape [num_samples, num_labels]
            - For multi-class task: class ID-valued array of shape [num_samples, 1]
        proportion_per_split (List[float]):
            Assume [train, test] if n_splits=2, or [train, val, test] if n_splits=3.
    Returns:
        tuple:
            - For n_splits=2: (trainval_indices, trainval_labels, test_indices, test_labels)
            - For n_splits=3: (train_indices, train_labels, val_indices, val_labels, test_indices, test_labels)
            - indices shape: [num_samples, 1], labels shape: [num_samples, 1] or [num_samples, num_labels]
    """
    assert len(proportion_per_split) == n_splits, f"Length of sample_distribution_per_split must be {n_splits}"
    assert isclose(sum(proportion_per_split), 1.0), "Sum of sample_distribution_per_split must be 1"

    indices = np.arange(labels.shape[0]).reshape(-1, 1)

    # ===========================================
    # First, shuffle for randomness
    # ===========================================
    perm = np.random.RandomState(seed).permutation(len(labels))
    labels = labels[perm]
    indices = indices[perm]

    # ===========================================
    # Second, trainval vs test
    # ===========================================
    test_prop = proportion_per_split[-1]

    trainval_indices, trainval_labels, test_indices, test_labels = deterministic_iterative_train_test_split(
        X=indices,
        y=labels,
        test_size=test_prop,
    )

    if n_splits == 2:
        return trainval_indices, trainval_labels, test_indices, test_labels

    # ===========================================
    # Third, train vs val
    # ===========================================
    val_prop = float(proportion_per_split[1]) / (proportion_per_split[0] + proportion_per_split[1])

    train_indices, train_labels, val_indices, val_labels = deterministic_iterative_train_test_split(
        X=trainval_indices,
        y=trainval_labels,
        test_size=val_prop,
    )

    return train_indices, train_labels, val_indices, val_labels, test_indices, test_labels


def plot_num_pos_labels_per_sample(
    *labels_per_split: List[np.ndarray],
    save_path: str = None,
    bar_width: float = 0.25,
    split_names: List[str] = None,
    title: str = "",
):
    """
    For multi-label task, each labels is a 0-1-valued array of shape [num_samples, num_labels].
    """
    if split_names:
        assert len(labels_per_split) == len(split_names), f"num splits: {len(labels_per_split)} != num split names: {len(split_names)}"
    else:
        split_names = [f"split_{i}" for i in range(len(labels_per_split))]

    for i in range(1, len(labels_per_split)):
        assert labels_per_split[i].shape[1] == labels_per_split[0].shape[1], f"num labels in split {i}: {labels_per_split[i].shape[1]} != num labels in split 0: {labels_per_split[0].shape[1]}"

    num_splits = len(labels_per_split)
    num_labels = labels_per_split[0].shape[1]

    num_pos_labels_per_sample_per_split = [np.sum(l, axis=1) for l in labels_per_split]

    xs = []  # per split number of positive labels
    ys = []  # per split number of samples having the corresponding number of positive labels
    for n in num_pos_labels_per_sample_per_split:
        unique, unique_counts = np.unique(n, return_counts=True)
        xs.append(unique)
        ys.append(unique_counts)
    normed_ys = [y / l.shape[0] for y, l in zip(ys, labels_per_split)]  # normalization

    plt.figure(
        figsize=(
            (num_labels + 1) * (num_splits * bar_width + 2 * bar_width),
            20 * bar_width,
        )
    )
    for i in range(num_splits):
        plt.bar(
            xs[i] + (i - (num_splits - 1) / 2.0) * bar_width,
            normed_ys[i],
            width=bar_width,
            label=f"{split_names[i]}: n={labels_per_split[i].shape[0]}",
            align="center",
        )
    plt.xticks(np.arange(num_labels + 1))
    plt.xlabel("Number of Positive Labels")
    plt.ylabel("Proportion of Samples")
    plt.title(title, fontweight="bold")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def plot_num_pos_samples_per_label(
    *labels_per_split: List[np.ndarray],
    save_path: str = None,
    bar_width: float = 0.25,
    split_names: List[str] = None,
    title: str = "",
):
    """
    For multi-label task, each labels is a 0-1-valued array of shape [num_samples, num_labels].
    """
    if split_names:
        assert len(labels_per_split) == len(split_names), f"num splits: {len(labels_per_split)} != num split names: {len(split_names)}"
    else:
        split_names = [f"split_{i}" for i in range(len(labels_per_split))]

    for i in range(1, len(labels_per_split)):
        assert labels_per_split[i].shape[1] == labels_per_split[0].shape[1], f"num labels in split {i}: {labels_per_split[i].shape[1]} != num labels in split 0: {labels_per_split[0].shape[1]}"

    num_splits = len(labels_per_split)
    num_labels = labels_per_split[0].shape[1]

    num_pos_samples_per_label_per_split = [np.sum(l, axis=0) for l in labels_per_split]

    x = np.arange(num_labels)  # label IDs
    normed_ys = [c / l.shape[0] for c, l in zip(num_pos_samples_per_label_per_split, labels_per_split)]  # normalized per split number of positive samples per label

    plt.figure(
        figsize=(
            num_labels * (num_splits * bar_width + 2 * bar_width),
            20 * bar_width,
        )
    )
    for i in range(num_splits):
        plt.bar(
            x + (i - (num_splits - 1) / 2.0) * bar_width,
            normed_ys[i],
            width=bar_width,
            label=f"{split_names[i]}: n={labels_per_split[i].shape[0]}",
            align="center",
        )
    plt.xticks(x)
    plt.xlabel("Label ID")
    plt.ylabel("Proportion of Positive Samples")
    plt.title(title, fontweight="bold")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def plot_num_samples_per_class(
    *labels_per_split: List[np.ndarray],
    save_path: str = None,
    bar_width: float = 0.25,
    split_names: List[str] = None,
    title: str = "",
):
    """
    For multi-class task, each labels is a class ID-valued array of shape [num_samples, 1].
    """
    if split_names:
        assert len(labels_per_split) == len(split_names), f"num splits: {len(labels_per_split)} != num split names: {len(split_names)}"
    else:
        split_names = [f"split_{i}" for i in range(len(labels_per_split))]

    num_splits = len(labels_per_split)
    num_classes = max([np.max(l) for l in labels_per_split]) + 1

    xs = []  # per split class IDs
    ys = []  # per split number of samples per class
    for l in labels_per_split:
        unique, unique_counts = np.unique(l, return_counts=True)
        xs.append(unique)
        ys.append(unique_counts)
    normed_ys = [y / l.shape[0] for y, l in zip(ys, labels_per_split)]  # normalization

    plt.figure(
        figsize=(
            # width = num_classes * (num_splits * bar_with + left_right_margin)
            num_classes * (num_splits * bar_width + 2 * bar_width),
            20 * bar_width,
        )
    )
    for i in range(num_splits):
        plt.bar(
            xs[i] + (i - (num_splits - 1) / 2.0) * bar_width,
            normed_ys[i],
            width=bar_width,
            label=f"{split_names[i]}: n={labels_per_split[i].shape[0]}",
            align="center",
        )
    plt.xticks(np.arange(num_classes))
    plt.xlabel("Class ID")
    plt.ylabel("Proportion of Samples")
    plt.title(title, fontweight="bold")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def generate_test_data(num_samples: int, num_labels: int, seed: int = 42):
        """
        Generate synthetic multi-label data for testing.

        Args:
            num_samples (int): The number of samples.
            num_labels (int): The number of labels.
            seed (int): Random seed for reproducibility.

        Returns:
            np.ndarray: A 0-1 valued array of shape [num_samples, num_labels].
        """
        np.random.seed(seed)
        return np.random.randint(0, 2, size=(num_samples, num_labels))

    def test_plot_functions():
        # Parameters for test data
        num_samples = 2000  # Number of samples
        num_labels = 10  # Number of labels
        n_splits = 3  # Number of splits (train, val, test)
        proportions = [0.7, 0.2, 0.1]  # Proportions for train, val, test

        # Generate synthetic multi-label data
        labels = generate_test_data(num_samples, num_labels)

        # Perform stratified sampling
        train_indices, train_labels, val_indices, val_labels, test_indices, test_labels = stratified_sampling(
            labels=labels,
            n_splits=n_splits,
            proportion_per_split=proportions,
            seed=2333,
        )

        # Call the plotting functions for visualization
        plot_num_pos_samples_per_label(
            train_labels,
            val_labels,
            test_labels,
            split_names=["Train", "Validation", "Test"],
            title="Proportion of Positive Samples Per Label",
            save_path="num_pos_samples_per_label_test.png",
        )

        plot_num_pos_labels_per_sample(
            train_labels,
            val_labels,
            test_labels,
            split_names=["Train", "Validation", "Test"],
            title="Proportion of Labels Per Sample",
            save_path="num_pos_labels_per_sample_test.png",
        )

    test_plot_functions()
