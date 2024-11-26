from bsai.src.domain.clusterer import HDBSCANClusterer, BaseClusterer
from bsai.src.domain.llm import OpenAIModel, BaseLLM
from bsai.src.domain.parser import TavilyParser, BaseParser
from bsai.src.domain.repository import BaseRepository, DFRepository
from bsai.src.domain.vectorizer import OpenAIVectorizer, BaseVectorizer

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
    return TavilyParser(settings.tavily.token)


def build_repository() -> BaseRepository:
    return DFRepository(settings.data.path)
