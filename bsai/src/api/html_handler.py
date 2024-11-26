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
from bsai.src.domain.core import pipeline_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    # if not os.path.exists(args.path):
    #     raise FileNotFoundError(f"Path {args.path} does not exist")

    # with open(args.path) as f:
    #     bookmarks = f.read()
    #
    # soup = BeautifulSoup(bookmarks, 'lxml')
    # links = soup.find_all('a')
    # hrefs = [l['href'] for l in links]

    hrefs = [
        "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms?nthPub=51",
        "https://hdbscan.readthedoces.io/en/latest/faq.html",
        "https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html",
        "https://medium.com/@garysvenson09/how-to-use-duckduckgo-search-with-python-in-langchain-157a816fa8d5",
        "https://ferd.ca/a-distributed-systems-reading-list.html",
        "https://imbue.com/research/70b-infrastructure/"
    ]

    url_parser = build_parser()
    vectorizer = build_vectorizer()
    clusterizer = build_clusterizer()
    llm = build_llm()
    repository = build_repository()
    pipeline_urls(hrefs, url_parser, vectorizer, clusterizer, llm, repository)


if __name__ == "__main__":
    main()