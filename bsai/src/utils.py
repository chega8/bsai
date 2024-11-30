import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

from bsai.src.types.dto import Cluster, Vector


def visualize_clusters(clusters: Cluster, vectors: Vector):
    matrix = np.array(vectors.vectors)
    df = pd.DataFrame(matrix)
    df["Cluster"] = clusters.labels
    df["Cluster"] = df["Cluster"] + 1

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    for category, color in enumerate(["purple", "green", "red"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified visualized in language 2d using t-SNE")
    plt.show()


def filter_urls(urls1: list[str], urls2: list[str]) -> set:
    return set(urls1).difference(set(urls2))
