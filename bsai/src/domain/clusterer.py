import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
import hdbscan


class BaseClusterer:
    def clusterize(self, vectors: list[list[float]]) -> list[int]:
        raise NotImplementedError

    def samples_from_cluster(
            self,
            vectors: list[np.ndarray],
            cluster_id: int,
            n_samples: int = 5
    ) -> np.ndarray:
        raise NotImplementedError

    def predict(self, vectors: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class HDBSCANClusterer(BaseClusterer):
    def __init__(self, min_cluster_size: int = 5):
        self.hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    def clusterize(self, vectors: list[np.ndarray]) -> list[int]:
        self.hdbs.fit(vectors)
        return self.hdbs.labels_

    def samples_from_cluster(
            self,
            vectors: list[np.ndarray],
            cluster_id: int,
            n_samples: int = 5
    ) -> np.ndarray:
        cluster_samples = [i for i, v in enumerate(vectors) if self.hdbs.labels_[i] == cluster_id]
        return np.random.choice(cluster_samples, n_samples)

    def predict(self, vectors: list[np.ndarray]) -> np.ndarray:
        labels, probs = hdbscan.approximate_predict(self.hdbs, vectors)
        return labels
