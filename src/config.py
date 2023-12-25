from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Base settings class."""

    postgres_dsn: PostgresDsn = "postgresql://postgres:postgres@localhost:5432/postgres"

    model_config = SettingsConfigDict(env_file="local.env")
