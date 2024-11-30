from typing import List

import numpy as np

from bsai.src.domain.clusterer import BaseClusterer
from bsai.src.domain.llm import BaseLLM
from bsai.src.domain.parser import BaseParser
from bsai.src.domain.repository import BaseRepository
from bsai.src.domain.vectorizer import BaseVectorizer

from loguru import logger

from bsai.src.types.dto import ParsedText, Summary, Vector, Cluster
from bsai.src.utils import visualize_clusters


def parse_urls(urls: list[str], parser: BaseParser) -> ParsedText:
    urls, texts = parser.extract(urls)
    return ParsedText(urls=urls, texts=texts)


def generate_summary(parsed_text: ParsedText, llm: BaseLLM) -> Summary:
    summary = llm.get_summary(parsed_text.texts)
    texts = []
    urls = []
    for url, s in zip(parsed_text.urls, summary):
        if len(s) > 0:
            urls.append(url)
            texts.append(s)
    return Summary(urls=urls, texts=texts)


def generate_embedding(summaries: Summary, vectorizer: BaseVectorizer) -> Vector:
    vectors = vectorizer.fit_transform(summaries.texts)
    return Vector(urls=summaries.urls, vectors=vectors)


def generate_clusters(
        vectors: Vector, clusterer: BaseClusterer
) -> Cluster:
    labels = clusterer.clusterize(vectors.vectors)
    return Cluster(urls=vectors.urls, labels=labels, texts=[])


def generate_topic(texts: list[str], llm: BaseLLM) -> str:
    return llm.get_cluster_topic(texts)


def clusters_to_summary(
        summaries: Summary,
        clusters: Cluster,
        clusterer: BaseClusterer,
        vectors: Vector,
        llm: BaseLLM,
) -> Cluster:
    cluster_to_summary = {}
    for cluster_id in set(clusters.labels):
        samples_idx = clusterer.samples_from_cluster(vectors.vectors, cluster_id)
        samples = [summaries.texts[i] for i in samples_idx]
        cluster_to_summary[cluster_id] = generate_topic(samples, llm)

    cluster_urls = []
    labels = []
    texts = []
    for cluster_id, cluster_url in zip(clusters.labels, clusters.urls):
        cluster_text = cluster_to_summary[cluster_id]
        cluster_urls.append(cluster_url)
        labels.append(cluster_id)
        texts.append(cluster_text)
    return Cluster(urls=cluster_urls, labels=labels, texts=texts)


def pipeline_urls(
        urls: list[str],
        parser: BaseParser,
        vectorizer: BaseVectorizer,
        clusterer: BaseClusterer,
        llm: BaseLLM,
        repository: BaseRepository,
):
    urls = repository.get_not_existing_urls(urls)
    n_urls = len(urls)
    if n_urls > 0:
        repository.save_urls(urls)
        logger.info(f"Found {len(urls)} new urls")

        # parsed = repository.get_texts()
        parsed = parse_urls(urls, parser)
        repository.save_texts(parsed)
        logger.info(f"Extracted {len(parsed.texts)} texts from {n_urls} urls")

        # summaries = repository.get_summaries()
        summaries = generate_summary(parsed, llm)
        repository.save_summaries(summaries)
        logger.info(f"Generated {len(summaries.texts)} summaries")

        # vectors = repository.get_vectors()
        vectors = generate_embedding(summaries, vectorizer)
        repository.save_vectors(vectors)
        logger.info(f"Generated {len(vectors.vectors)} embeddings")

        clusters = generate_clusters(vectors, clusterer)
        labels = clusters.labels
        logger.info(f"Clustering {len(labels)} points, {len(set(labels))} unique clusters")

        clusters = clusters_to_summary(summaries, clusters, clusterer, vectors, llm)
        repository.save_clusters(clusters)
        logger.info(f"Generated {len(set(clusters.texts))} cluster texts for {len(clusters.labels)} points")

        repository.save(parsed, summaries, vectors, clusters)


def recommend_random(repository: BaseRepository) -> tuple[str, str]:
    summary = repository.get_summaries()
    idx = np.random.choice(len(summary.urls))
    return summary.urls[idx], summary.texts[idx]


def show_clusters(repository: BaseRepository):
    visualize_clusters(repository.get_clusters(), repository.get_vectors())
