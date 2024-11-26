"""Fixtures and etc."""
from bsai.src.domain.llm import BaseLLM
from bsai.src.domain.parser import BaseParser
from bsai.src.domain.vectorizer import BaseVectorizer


class MockLLM(BaseLLM):
    def __init__(self):
        pass

    def generate(self, query: str) -> str:
        return "response"

    def get_summary(self, texts: list[str]) -> list[str]:
        return [f"summary_{i}" for i in range(len(texts))]

    def get_cluster_topic(self, texts: list[str]) -> str:
        return "topic"


class MockParser(BaseParser):
    def __init__(self):
        pass

    def extract(self, urls: list[str]) -> list[str]:
        return [f"extracted_text_{i}" for i in range(len(urls))]


class MocVectorizer(BaseVectorizer):
    def __init__(self):
        pass

    def transform(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 1.0, 1.0, 1.0] for _ in texts]

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        return self.transform(texts)
