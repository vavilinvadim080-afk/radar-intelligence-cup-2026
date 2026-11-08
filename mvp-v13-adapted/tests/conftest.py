import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest


@pytest.fixture(autouse=True)
def _reset_post_rate_limiter():
    """Лимиты частоты (POST, кнопки, чтение) общие на процесс: между тестами обнуляем, чтобы длинный прогон не упёрся в 429."""
    import sys
    srv = sys.modules.get("app.server")
    if srv is not None:
        for name in ("rate_limiter", "button_limiter", "read_limiter"):
            if hasattr(srv, name):
                getattr(srv, name).reset()
    yield
