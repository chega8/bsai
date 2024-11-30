import ast
import os
from abc import ABC
from typing import Any

import pandas as pd
from bsai.src.types.dto import ParsedText, Summary, Vector, Cluster
from loguru import logger

class BaseRepository(ABC):
    def __init__(self):
        ...

    def save(
            self,
            parsed: ParsedText,
            summaries: Summary,
            vectors: Vector,
            clusters: Cluster,
    ):
        raise NotImplementedError

    def save_texts(self, parsed_text: ParsedText):
        raise NotImplementedError

    def save_summaries(self, summary: Summary):
        raise NotImplementedError

    def get_summaries(self) -> Summary:
        raise NotImplementedError

    def save_vectors(self, vectors: Vector):
        raise NotImplementedError

    def get_vectors(self) -> Vector:
        raise NotImplementedError

    def save_clusters(self, clusters: Cluster):
        raise NotImplementedError

    def save_cluster_texts(self, urls: list[str], cluster_texts: list[str]):
        raise NotImplementedError

    def get(self, url: str | None, cluster_id: int | None) -> pd.DataFrame:
        raise NotImplementedError

    def get_texts(self) -> ParsedText:
        raise NotImplementedError

    def get_clusters(self) -> Cluster:
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

    def _get(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save(
            self,
            parsed: ParsedText,
            summaries: Summary,
            vectors: Vector,
            clusters: Cluster,
    ):
        union_urls = set(parsed.urls) & set(summaries.urls) & set(vectors.urls) & set(clusters.urls)
        logger.info(f"Saving {len(union_urls)} urls")
        urls = []
        texts = []
        summary = []
        vector = []
        labels = []
        cluster_texts = []
        for url in union_urls:
            urls.append(url)
            texts.append(parsed.texts[parsed.urls.index(url)])
            summary.append(summaries.texts[summaries.urls.index(url)])
            vector.append(vectors.vectors[vectors.urls.index(url)])
            labels.append(clusters.labels[clusters.urls.index(url)])
            cluster_texts.append(clusters.texts[clusters.urls.index(url)])

        self._save(
            os.path.join(self.path, "all.csv"),
            url=urls,
            text=texts,
            summary=summary,
            vector=vector,
            label=labels,
            cluster_text=cluster_texts,
        )

    def save_texts(self, parsed_text: ParsedText):
        path = os.path.join(self.path, "text.csv")
        self._save(path, url=parsed_text.urls, text=parsed_text.texts)

    def save_summaries(self, summary: Summary):
        path = os.path.join(self.path, "summary.csv")
        self._save(path, url=summary.urls, summary=summary.texts)

    def save_vectors(self, vectors: Vector):
        path = os.path.join(self.path, "vector.csv")
        self._save(path, url=vectors.urls, vector=vectors.vectors)

    def save_clusters(self, clusters: Cluster):
        path = os.path.join(self.path, "cluster.csv")
        self._save(path, url=clusters.urls, label=clusters.labels, cluster_text=clusters.texts)

    def get(self, url: str | None, cluster_id: int | None) -> pd.DataFrame:
        path = os.path.join(self.path, "all.csv")
        df = pd.read_csv(self.path)
        if url:
            return df[df['url'] == url]
        if cluster_id:
            return df[df['cluster_label'] == cluster_id]
        return df

    def get_texts(self) -> ParsedText:
        if not self.exist():
            return ParsedText(urls=[], texts=[])
        path = os.path.join(self.path, "text.csv")
        df = pd.read_csv(path)
        return ParsedText(urls=df['url'].tolist(), texts=df['text'].tolist())

    def get_summaries(self) -> Summary:
        path = os.path.join(self.path, "summary.csv")
        summaries = self._get(path)
        return Summary(urls=summaries['url'].tolist(), texts=summaries['summary'].tolist())

    def get_vectors(self) -> Vector:
        path = os.path.join(self.path, "vector.csv")
        vectors = self._get(path)
        vectors['vector'] = vectors['vector'].apply(ast.literal_eval)
        return Vector(urls=vectors['url'].tolist(), vectors=vectors['vector'].tolist())

    def get_clusters(self) -> Cluster:
        if not self.exist():
            return Cluster(urls=[], labels=[], texts=[])

        path = os.path.join(self.path, "cluster.csv")
        df = self._get(path)
        return Cluster(
            urls=df['url'].tolist(),
            labels=df['label'].tolist(),
            texts=df['cluster_text'].tolist()
        )

    def exist(self) -> bool:
        return os.path.exists(os.path.join(self.path, "text.csv"))
