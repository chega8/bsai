import argparse
import os

from bs4 import BeautifulSoup

from bsai.src.dependency import (
    build_clusterizer,
    build_llm,
    build_parser,
    build_repository,
    build_vectorizer,
)
from bsai.src.domain.core import pipeline_urls, recommend_random
from loguru import logger

from bsai.src.utils import visualize_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Path {args.path} does not exist")

    with open(args.path) as f:
        bookmarks = f.read()

    soup = BeautifulSoup(bookmarks, 'html.parser')
    links = soup.find_all('a')
    hrefs = [l['href'] for l in links]

    url_parser = build_parser()
    vectorizer = build_vectorizer()
    clusterizer = build_clusterizer()
    llm = build_llm()
    repository = build_repository()
    pipeline_urls(hrefs, url_parser, vectorizer, clusterizer, llm, repository)

    visualize_clusters(repository.get_clusters(), repository.get_vectors())
    url, text = recommend_random(repository)
    logger.info(f"Recommended URL: {url}, summary: {text}")


if __name__ == "__main__":
    main()
