import pytest
from src.core.config import config


def test_config_defaults():
    assert config.PROJECT_NAME.startswith("LexiRAG")
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.TOP_K_RESULTS >= 1
    assert "http://localhost:3000" in config.ALLOWED_ORIGINS
