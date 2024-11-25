from typing import List

from bsai.src.domain.clusterizer import BaseClusterizer
from bsai.src.domain.llm import BaseLLM
from bsai.src.domain.parser import BaseParser
from bsai.src.domain.repository import BaseRepository
from bsai.src.domain.vectorizer import BaseVectorizer


def parse_urls(urls: list[str], parser: BaseParser) -> list[str]:
    return parser.extract(urls)


def generate_clusters(vectors: list[list[float]], clusterizer: BaseClusterizer) -> List[int]:
    return clusterizer.clusterize(vectors)


def generate_summary(text: list[str], llm: BaseLLM) -> list[str]:
    return llm.get_summary(text)


def generate_topic(texts: list[str], llm: BaseLLM) -> str:
    return llm.get_cluster_topic(texts)


def generate_embedding(text, vectorizer: BaseVectorizer) -> list[list[float]]:
    return vectorizer.transform(text)


def clusters_to_summary(
        summaries: list[str],
        clusters: list[int],
        clusterizer: BaseClusterizer,
        vectors,
        llm: BaseLLM
) -> list[str]:
    cluster_to_summary = {}
    for cluster_id in set(clusters):
        samples_idx = clusterizer.samples_from_cluster(vectors, cluster_id)
        samples = [summaries[i] for i in samples_idx]
        cluster_to_summary[cluster_id] = generate_topic(samples, llm)

    return [cluster_to_summary[cluster_id] for cluster_id in clusters]


def pipeline_urls(
        urls: list[str],
        parser: BaseParser,
        vectorizer: BaseVectorizer,
        clusterizer: BaseClusterizer,
        llm: BaseLLM,
        repository: BaseRepository
):
    existing_urls = set(repository.get_urls())
    urls = [url for url in urls if url not in existing_urls]

    texts = parse_urls(urls, parser)
    vectors = generate_embedding(texts, vectorizer)
    clusters = generate_clusters(vectors, clusterizer)
    summaries = generate_summary(texts, llm)
    cluster_text = clusters_to_summary(summaries, clusters, clusterizer, vectors, llm)
    repository.save(urls, texts, summaries, vectors, clusters, cluster_text)


def handle_new_urls(
        urls: list[str],
        parser: BaseParser,
        vectorizer: BaseVectorizer,
        clusterizer: BaseClusterizer,
        llm: BaseLLM,
        repository: BaseRepository
):
    exisiting_urls = set(repository.get_urls())
    urls = [url for url in urls if url not in exisiting_urls]
