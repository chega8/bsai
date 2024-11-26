import os
from abc import ABC
from typing import Any
import pandas as pd


class BaseRepository(ABC):
    def __init__(self):
        ...

    def save(
            self,
            urls: list[str],
            texts: list[str],
            summary: list[str],
            vectors: list[list[float]],
            cluster_labels: list[int],
            cluster_texts: list[str]
    ):
        raise NotImplementedError

    def save_text(self, urls: list[str], texts: list[str]):
        raise NotImplementedError

    def save_summaries(self, urls: list[str], summaries: list[str]):
        raise NotImplementedError

    def save_vectors(self, urls: list[str], vectors: list[list[float]]):
        raise NotImplementedError

    def save_clusters(self, urls: list[str], clusters: list[int]):
        raise NotImplementedError

    def save_cluster_texts(self, urls: list[str], cluster_texts: list[str]):
        raise NotImplementedError

    def get(self, url: str | None, cluster_id: int | None) -> Any:
        raise NotImplementedError

    def get_urls(self):
        raise NotImplementedError

    def get_clusters(self):
        raise NotImplementedError

    def exist(self) -> bool:
        raise NotImplementedError


class DFRepository(BaseRepository):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def _save(self, path, **kwargs):
        data = pd.DataFrame(kwargs)
        columns = data.columns
        data.to_csv(path, mode='a', header=columns if not os.path.exists(path) else False, index=False)
        del data

    def save(
            self,
            urls: list[str],
            texts: list[str],
            summary: list[str],
            vectors: list[list[float]],
            cluster_labels: list[int],
            cluster_texts: list[str]
    ):
        self._save(
            url=urls,
            text=texts,
            summary=summary,
            vector=vectors,
            cluster_label=cluster_labels,
            cluster_text=cluster_texts
        )

    def save_text(self, urls: list[str], texts: list[str]):
        path = self.path + "_text.csv"
        self._save(path, url=urls, text=texts)

    def save_summaries(self, urls: list[str], summaries: list[str]):
        path = self.path + "_summary.csv"
        self._save(path, url=urls, summary=summaries)

    def save_vectors(self, urls: list[str], vectors: list[list[float]]):
        path = self.path + "_vector.csv"
        self._save(path, url=urls, vector=vectors)

    def save_clusters(self, urls: list[str], clusters: list[int]):
        path = self.path + "_cluster.csv"
        self._save(path, url=urls, cluster_label=clusters)

    def save_cluster_texts(self, urls: list[str], cluster_texts: list[str]):
        path = self.path + "_cluster_text.csv"
        self._save(path, url=urls, cluster_text=cluster_texts)

    def get(self, url: str | None, cluster_id: int | None) -> Any:
        df = pd.read_csv(self.path)
        if url:
            return df[df['url'] == url]
        if cluster_id:
            return df[df['cluster_label'] == cluster_id]
        return df

    def get_urls(self):
        if not self.exist():
            return []
        df = pd.read_csv(self.path)
        return df['url'].tolist()

    def get_clusters(self):
        if not self.exist():
            return []
        df = pd.read_csv(self.path)
        return df['cluster_text'].unique().tolist()

    def exist(self) -> bool:
        return os.path.exists(self.path)
