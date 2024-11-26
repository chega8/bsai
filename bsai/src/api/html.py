import argparse
import os

from bs4 import BeautifulSoup

import bsai
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

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Path {args.path} does not exist")

    with open(args.path) as f:
        bookmarks = f.read()

    soup = BeautifulSoup(bookmarks, 'lxml')
    links = soup.find_all('a')
    hrefs = [l['href'] for l in links]

    url_parser = build_parser()
    vectorizer = build_vectorizer()
    clusterizer = build_clusterizer()
    llm = build_llm()
    repository = build_repository()
    pipeline_urls(hrefs, url_parser, vectorizer, clusterizer, llm, repository)


if __name__ == "__main__":
    print("Hello, World!")
