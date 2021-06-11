import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import seaborn as sns
import pandas as pd
import os
import numpy as np

# HE_VIEWS = ["LA", "KAKL", "KAPAP", "KAAP", "CV"]
HE_VIEWS = ["WHITE", "BLACK", "ASIAN", "INDIAN", "OTHER"]


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, help="Name used for charts", required=True)
    parser.add_argument("--log_dir", type=str, help="FQ path to embedding logs", required=True)

    args = parser.parse_args()
    return args


def load_data_from_fq_path(fq_path):
    fq_path = os.path.expanduser(fq_path)
    return tf.train.load_variable(fq_path, "embedding")


def load_labels_from_fq_path(fq_path):
    fq_path = os.path.expanduser(fq_path)
    return pd.read_table("{}/{}".format(fq_path, "metadata.tsv"), sep="\t", names=["labels"])


def load_results_data(fq_path):
    fq_path = os.path.expanduser(fq_path)
    return np.load("{}/conf_mat.npy".format(fq_path))


def main():
    # Parse arguments
    args = parse_args()

    # PCA
    data = load_data_from_fq_path(args.log_dir)
    standardized_data = StandardScaler().fit_transform(data)
    pca_obj = PCA(2)
    pca_obj.fit(standardized_data)
    pca_data = pca_obj.transform(standardized_data)
    labels = load_labels_from_fq_path(args.log_dir)["labels"]
    label_map = ["Male", "Female"]
    # label_map = ["White", "Black", "Asian", "Indian"]
    labels = labels.apply(lambda x: label_map[x])

    df = pd.DataFrame(
        {"PC1": pca_data[:, 0], "PC2": pca_data[:, 1], "Label": labels})

    figure = plt.figure(figsize=(8, 10))
    figure.set_rasterized(True)
    # colours = sns.color_palette("tab10", len(labels.unique()))
    colours = [sns.color_palette("tab10", 10)[8], sns.color_palette("tab10", 10)[9]]
    ax = sns.scatterplot(x="PC1", y="PC2", hue="Label", data=df, legend="full", alpha=0.3, palette=colours)
    plt.title(args.name, fontsize=18)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # plt.figtext(0.01, 0.01, 'Explained Variance {:.4f}'.format(np.sum(pca_obj.explained_variance_ratio_)))
    ax.set_rasterized(True)
    plt.savefig(os.path.expanduser(args.log_dir) + "/PCA.eps", rasterized=True)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 8))
    conf_mat = load_results_data(args.log_dir)
    ax = sns.heatmap(conf_mat, cmap="YlGnBu", annot=True, fmt="d")
    ax.set_xticklabels(HE_VIEWS)
    ax.set_yticklabels(HE_VIEWS)
    ax.set(xlabel="Predicted Cluster", ylabel="True Cluster")
    plt.title(args.name + " confusion matrix")
    plt.savefig(os.path.expanduser(args.log_dir) + "/confusion_matrix.eps")
    plt.close()


if __name__ == '__main__':
    main()
