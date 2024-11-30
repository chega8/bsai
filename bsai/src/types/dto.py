import numpy as np
from pydantic import BaseModel


class ParsedText(BaseModel):
    urls: list[str]
    texts: list[str]


class Summary(BaseModel):
    urls: list[str]
    texts: list[str]


class Vector(BaseModel):
    urls: list[str]
    vectors: list[list[float]]


class Cluster(BaseModel):
    urls: list[str]
    labels: list[int]
    texts: list[str]
