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
        cluster_texts: list[str],
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

    def get(self, url: str | None, cluster_id: int | None) -> pd.DataFrame:
        raise NotImplementedError

    def get_urls(self) -> tuple[list[str], list[str]]:
        raise NotImplementedError

    def get_clusters(self) -> list[int]:
        raise NotImplementedError

    def exist(self) -> bool:
        raise NotImplementedError


class DFRepository(BaseRepository):
    def __init__(self, path: str):
        super().__init__()
        self.path = os.path.join(path, "df_storage")
        os.makedirs(self.path, exist_ok=True)

    def _save(self, path, **kwargs):
        data = pd.DataFrame(kwargs)
        columns = data.columns
        data.to_csv(
            path,
            mode='a',
            header=columns if not os.path.exists(path) else False,
            index=False,
        )
        del data

    def save(
        self,
        urls: list[str],
        texts: list[str],
        summary: list[str],
        vectors: list[list[float]],
        cluster_labels: list[int],
        cluster_texts: list[str],
    ):
        self._save(
            os.path.join(self.path, "all.csv"),
            url=urls,
            text=texts,
            summary=summary,
            vector=vectors,
            cluster_label=cluster_labels,
            cluster_text=cluster_texts,
        )

    def save_text(self, urls: list[str], texts: list[str]):
        path = os.path.join(self.path, "text.csv")
        self._save(path, url=urls, text=texts)

    def save_summaries(self, urls: list[str], summaries: list[str]):
        path = os.path.join(self.path, "summary.csv")
        self._save(path, url=urls, summary=summaries)

    def save_vectors(self, urls: list[str], vectors: list[list[float]]):
        path = os.path.join(self.path, "vector.csv")
        self._save(path, url=urls, vector=vectors)

    def save_clusters(self, urls: list[str], clusters: list[int]):
        path = os.path.join(self.path, "cluster.csv")
        self._save(path, url=urls, cluster_label=clusters)

    def save_cluster_texts(self, urls: list[str], cluster_texts: list[str]):
        path = os.path.join(self.path, "cluster_text.csv")
        self._save(path, url=urls, cluster_text=cluster_texts)

    def get(self, url: str | None, cluster_id: int | None) -> pd.DataFrame:
        path = os.path.join(self.path, "all.csv")
        df = pd.read_csv(self.path)
        if url:
            return df[df['url'] == url]
        if cluster_id:
            return df[df['cluster_label'] == cluster_id]
        return df

    def get_urls(self) -> tuple[list[str], list[str]]:
        if not self.exist():
            return [], []
        path = os.path.join(self.path, "text.csv")
        df = pd.read_csv(path)
        return df['url'].tolist(), df['text'].tolist()

    def get_clusters(self) -> list[int]:
        if not self.exist():
            return []
        df = pd.read_csv(self.path)
        return df['cluster_text'].unique().tolist()

    def exist(self) -> bool:
        return os.path.exists(os.path.join(self.path, "text.csv"))
