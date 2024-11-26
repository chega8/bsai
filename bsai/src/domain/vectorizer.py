from abc import ABC

from bsai.src.domain.llm import OpenAIModel


class BaseVectorizer(ABC):
    def __init__(self):
        pass

    def transform(self, texts: list[str]) -> list[list[float]]:
        pass

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        pass


class OpenAIVectorizer(BaseVectorizer):
    def __init__(self, emb_model='text-embedding-3-large'):
        super().__init__()
        self.llm = OpenAIModel('gpt-4o')
        self.emb_model = emb_model

    def transform(self, texts: list[str]) -> list[list[float]]:
        return [self.llm.get_embedding(text, self.emb_model) for text in texts]

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        return self.transform(texts)
