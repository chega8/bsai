from loguru import logger
from openai import OpenAI

from config import settings


class BaseLLM:
    def generate(self, query: str) -> str:
        raise NotImplementedError

    def get_summary(self, text: list[str]) -> list[str]:
        raise NotImplementedError

    def get_cluster_topic(self, texts: list[str]) -> str:
        raise NotImplementedError


class OpenAIModel(BaseLLM):
    def __init__(self, name: str):
        self.name = name
        self.client = OpenAI(api_key=settings.llm.token)

    def generate(self, query: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                messages=[{'role': 'user', 'content': query}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(e)
            return ""

    def get_summary(self, texts: list[str]) -> list[str]:
        results = []
        for text in texts:
            results.append(self.get_single_summary(text))
        return results

    def get_single_summary(self, text: str) -> str:
        prompt = """Summarize the following text:
        {}
        
        Summary:""".format(
            text
        )
        return self.generate(prompt)

    def get_cluster_topic(self, texts: list[str]) -> str:
        cluster_text = "\n".join(texts)
        prompt = """Extract the topic from the following texts:
        {}
        Topic:""".format(
            cluster_text
        )
        return self.generate(prompt)

    def get_embedding(self, text: str, model="text-embedding-3-large"):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=model).data[0].embedding
        )
