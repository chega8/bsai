from abc import ABC, abstractmethod

from tavily import TavilyClient
from loguru import logger

class BaseParser(ABC):
    @abstractmethod
    def extract(self, urls: list[str]) -> tuple[list[str], list[str]]:
        pass


class TavilyParser(BaseParser):
    def __init__(self, token: str):
        self.client = TavilyClient(api_key=token)

    def extract(self, urls: list[str], chunk_size: int = 2) -> tuple[list[str], list[str]]:
        result = []
        valid_urls = []
        for i in range(0, len(urls), chunk_size):
            chunk = urls[i:i + chunk_size]
            chunk_result, chunk_valid_urls = self._extract(chunk)
            result.extend(chunk_result)
            valid_urls.extend(chunk_valid_urls)
        return result, valid_urls

    def _extract(self, urls: list[str]) -> tuple[list[str], list[str]]:
        result = []
        valid_urls = []
        try:
            results = self.client.extract(urls=urls)["results"]
            for content in results:
                result.append(content["raw_content"])
                valid_urls.append(content["url"])
        except Exception as e:
            logger.error(e)

        return result, valid_urls
