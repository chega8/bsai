from bsai.src.domain.clusterer import BaseClusterer, HDBSCANClusterer
from bsai.src.domain.llm import BaseLLM, OpenAIModel
from bsai.src.domain.parser import BaseParser, TavilyParser
from bsai.src.domain.repository import BaseRepository, DFRepository
from bsai.src.domain.vectorizer import BaseVectorizer, OpenAIVectorizer
from config import settings


def build_llm() -> BaseLLM:
    return OpenAIModel(settings.llm.model)


def build_recommender():
    ...


def build_vectorizer() -> BaseVectorizer:
    return OpenAIVectorizer()


def build_clusterizer() -> BaseClusterer:
    return HDBSCANClusterer()


def build_parser() -> BaseParser:
    return TavilyParser(settings.tavily.api_key)


def build_repository() -> BaseRepository:
    return DFRepository(settings.data.path)
