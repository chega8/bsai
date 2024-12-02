import os
from enum import Enum

from pydantic import AnyUrl, Extra, root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    host: str
    port: int
    user: str
    password: str
    db: str
    dsn: AnyUrl = None

    @root_validator(pre=True, skip_on_failure=True, allow_reuse=True)
    def init_postgres_dsn(cls, values):
        values[
            "dsn"
        ] = f"postgresql://{values['user']}:{values['password']}@{values['host']}:{values['port']}/{values['db']}"
        return values

    model_config = SettingsConfigDict(
        env_file="env", env_file_encoding="utf-8", env_prefix="PG_", extra="allow"
    )


class TavilySettings(BaseSettings):
    api_key: str

    model_config = SettingsConfigDict(
        env_file="env", env_file_encoding="utf-8", env_prefix="TAVILY_", extra="allow"
    )


class LLMSettings(BaseSettings):
    model: str
    token: str

    model_config = SettingsConfigDict(
        env_file="env", env_file_encoding="utf-8", env_prefix="LLM_", extra="allow"
    )


class DataSettings(BaseSettings):
    path: str

    model_config = SettingsConfigDict(
        env_file="env", env_file_encoding="utf-8", env_prefix="DATA_", extra="allow"
    )


class Settings(BaseSettings):
    tavily: TavilySettings = TavilySettings()
    llm: LLMSettings = LLMSettings()
    postgres: PostgresSettings = PostgresSettings()
    data: DataSettings = DataSettings()


settings = Settings()
