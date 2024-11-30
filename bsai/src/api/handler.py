"""API Handler Module"""
import numpy as np

from bsai.src.domain.repository import BaseRepository


def recommend_random(repository: BaseRepository) -> tuple[str, str]:
    """Recommend random cluster"""
    clusters = repository.get_clusters()
    idx = np.random.choice(len(clusters.labels))
    return clusters.urls[idx], clusters.texts[idx]

