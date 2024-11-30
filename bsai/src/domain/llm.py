from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from config import settings

LONG_SUMMARY_PROMPT_SYSTEM = """You are an advanced summarization assistant. 
Your task is to generate complete, detailed, and informative summaries of the content from a website. 
These summaries should be factual and preserve the key information and details. 
Avoid including subjective opinions unless they are explicitly stated in the source content."""

LONG_SUMMARY_PROMPT_USER = """I will provide the content or description of a webpage. Summarize it as follows:

- Extract the most important key points and concepts from the content.
- Ensure the summary captures the main idea, subtopics, and specific details without leaving out crucial information.
- Organize the summary in a clear and coherent manner.

If the content contains useless or irrelevant information, for example, advertisements, please ignore it.
Also if content contains errors or login page text, please ignore it. Mark the content as "useful": False.

Input:
{query}

Summary:
"""

SHORT_SUMMARY_PROMPT_SYSTEM = """You are a clustering assistant specializing in creating short, meaningful summaries. 
Your task is to analyze few detailed summaries within a cluster and generate a brief, highly descriptive summary 
that captures the semantic core of the cluster. 
It must be concise yet semantically representative."""
SHORT_SUMMARY_PROMPT_USER = """I will provide you with 3-4 detailed summaries representing a cluster. Your task is to:

Identify the common theme or main idea shared across all summaries.
Generate a short and precise summary (1-2 words) that reflects the semantic essence of the cluster.

Input:
{query}

Cluster name:
"""


class Summary(BaseModel):
    is_useful: bool
    summary: str


class BaseLLM:
    def generate(self, query: str, structure: BaseModel | None = None) -> str:
        raise NotImplementedError

    def get_summary(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    def get_cluster_topic(self, texts: list[str]) -> str:
        raise NotImplementedError


class OpenAIModel(BaseLLM):
    def __init__(self, name: str):
        self.name = name
        self.client = OpenAI(api_key=settings.llm.token)

    def generate(self, dialog: list[dict], structure: BaseModel | None = None) -> str:
        try:
            if structure is None:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=dialog,
                )
                return completion.choices[0].message.content
            else:
                completion = self.client.beta.chat.completions.parse(
                    model=self.name,
                    messages=dialog,
                    response_format=Summary,
                )
                content = completion.choices[0].message.parsed
                if content.is_useful:
                    return content.summary
                else:
                    return ""
        except Exception as e:
            logger.error(e)
            return ""

    def get_summary(self, texts: list[str]) -> list[str]:
        results = []
        for query in texts:
            summary = self.get_single_summary(query)
            results.append(summary)
        return results

    def get_single_summary(self, query: str) -> str:
        dialog = [
            {'role': 'system', 'content': LONG_SUMMARY_PROMPT_SYSTEM},
            {'role': 'user', 'content': LONG_SUMMARY_PROMPT_USER.format(query=query)},
        ]
        return self.generate(dialog, Summary)

    def get_cluster_topic(self, texts: list[str]) -> str:
        cluster_texts = "\n- " + "\n- ".join(texts)
        dialog = [
            {'role': 'system', 'content': SHORT_SUMMARY_PROMPT_SYSTEM},
            {'role': 'user', 'content': SHORT_SUMMARY_PROMPT_USER.format(query=cluster_texts)},
        ]
        return self.generate(dialog)

    def get_embeddings(self, texts: list[str], model="text-embedding-3-large"):
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.client.embeddings.create(input=texts, model=model)
        return [emb.embedding for emb in embeddings.data]
