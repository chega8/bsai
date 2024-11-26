from abc import ABC, abstractmethod

from tavily import TavilyClient


class BaseParser(ABC):
    @abstractmethod
    def extract(self, urls: list[str]) -> list[str]:
        pass


class TavilyParser(BaseParser):
    def __init__(self, token: str):
        self.client = TavilyClient(api_key=token)

    def extract(self, urls: list[str]) -> tuple[list[str], list[str]]:
        result = []
        valid_urls = []
        for content in self.client.extract(urls=urls)["results"]:
            try:
                result.append(content["raw_content"])
                valid_urls.append(content["url"])
            except Exception as e:
                pass
        return result, valid_urls
