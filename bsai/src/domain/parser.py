from abc import ABC, abstractmethod

from tavily import TavilyClient


class BaseParser(ABC):
    @abstractmethod
    def extract(self, urls: list[str]) -> list[str]:
        pass


class TavilyParser(BaseParser):
    def __init__(self, token: str):
        self.client = TavilyClient(api_key=token)

    def extract(self, urls: list[str]) -> list[str]:
        result = []
        for content in self.client.extract(urls=urls)["results"]:
            try:
                result.append(content["raw_content"])
            except Exception as e:
                pass
        return result
