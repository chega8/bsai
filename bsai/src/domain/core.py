from typing import List

import numpy as np

from bsai.src.domain.clusterer import BaseClusterer
from bsai.src.domain.llm import BaseLLM
from bsai.src.domain.parser import BaseParser
from bsai.src.domain.repository import BaseRepository
from bsai.src.domain.vectorizer import BaseVectorizer


def parse_urls(urls: list[str], parser: BaseParser) -> list[str]:
    return parser.extract(urls)


def generate_clusters(vectors: list[list[float]], clusterer: BaseClusterer) -> List[int]:
    return clusterer.clusterize(vectors)


def generate_summary(texts: list[str], llm: BaseLLM) -> list[str]:
    return llm.get_summary(texts)


def generate_topic(texts: list[str], llm: BaseLLM) -> str:
    return llm.get_cluster_topic(texts)


def generate_embedding(text, vectorizer: BaseVectorizer) -> list[list[float]]:
    return vectorizer.transform(text)


def clusters_to_summary(
        summaries: list[str],
        clusters: list[int],
        clusterer: BaseClusterer,
        vectors,
        llm: BaseLLM
) -> list[str]:
    cluster_to_summary = {}
    for cluster_id in set(clusters):
        samples_idx = clusterer.samples_from_cluster(vectors, cluster_id)
        samples = [summaries[i] for i in samples_idx]
        cluster_to_summary[cluster_id] = generate_topic(samples, llm)

    return [cluster_to_summary[cluster_id] for cluster_id in clusters]


def pipeline_urls(
        urls: list[str],
        parser: BaseParser,
        vectorizer: BaseVectorizer,
        clusterer: BaseClusterer,
        llm: BaseLLM,
        repository: BaseRepository
):
    existing_urls = set(repository.get_urls())
    urls = [url for url in urls if url not in existing_urls]

    texts = parse_urls(urls, parser)
    repository.save_text(urls, texts)

    summaries = generate_summary(texts, llm)
    repository.save_summaries(urls, summaries)

    vectors = generate_embedding(texts, vectorizer)
    repository.save_vectors(urls, vectors)

    clusters = generate_clusters(vectors, clusterer)
    repository.save_clusters(urls, clusters)

    cluster_text = clusters_to_summary(summaries, clusters, clusterer, vectors, llm)
    repository.save_cluster_texts(urls, cluster_text)

    repository.save(urls, texts, summaries, vectors, clusters, cluster_text)


def recommend_random(repository: BaseRepository) -> tuple[str, str]:
    existing_urls = repository.get_urls()
    url = np.random.choice(existing_urls, 1)
    row = repository.get(url[0])

    return row['url'], row['summary']


def show_cluster(repository: BaseRepository, cluster_id: int) -> List[str]:
    rows = repository.get(cluster_id=cluster_id)
    return rows['url']
