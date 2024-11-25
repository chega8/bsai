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

    def get(self, url: str | None, cluster_id: int | None) -> Any:
        raise NotImplementedError

    def get_urls(self):
        raise NotImplementedError

    def exist(self) -> bool:
        raise NotImplementedError


class DFRepository(BaseRepository):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def save(
            self,
            urls: list[str],
            texts: list[str],
            summary: list[str],
            vectors: list[list[float]],
            cluster_labels: list[int],
            cluster_texts: list[str]
    ):
        rows = {
            'url': urls,
            'text': texts,
            'summary': summary,
            'vector': vectors,
            'cluster_label': cluster_labels,
            'cluster_text': cluster_texts
        }

        data = pd.DataFrame(rows)
        data.to_csv(self.path, index=False)
        del data
        del rows

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

    def exist(self) -> bool:
        return os.path.exists(self.path)