"""Защита демо: подпись записанного прогона (HMAC-SHA256), токены с ролями, лимит частоты POST.

Всё здесь без FastAPI, чтобы тесты вызывали функции напрямую. Подключение к HTTP в app/server.py.
Ограничение, названное прямо: HMAC защищает от подмены файла тем, кто НЕ знает ключ. Тот, кто читает диск целиком (файл ключа лежит рядом),
подпишет что угодно; в проде ключ берётся из RADAR_HMAC_KEY (менеджер секретов), а не из файла.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path


class ReplayTamperedError(Exception):
    """demo/replay.json не прошёл проверку подписи (изменён, подписан другим ключом или без подписи)."""


# ---------------------------------------------------------------- HMAC-подпись файла
def get_key(key_file: Path) -> tuple[bytes, bool]:
    """Ключ из RADAR_HMAC_KEY; иначе из key_file; иначе генерируется (32 случайных байта) и пишется в key_file (права 600).
    Возвращает (ключ, был_создан_сейчас)."""
    env = os.environ.get("RADAR_HMAC_KEY")
    if env:
        return env.encode("utf-8"), False
    key_file = Path(key_file)
    if key_file.exists():
        return bytes.fromhex(key_file.read_text(encoding="utf-8").strip()), False
    key = secrets.token_bytes(32)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key.hex() + "\n")
    return key, True


def _sig_path(path: Path) -> Path:
    return Path(str(path) + ".sig")


def sign_file(path: Path, key: bytes) -> str:
    path = Path(path)
    sig = hmac.new(key, path.read_bytes(), hashlib.sha256).hexdigest()
    _sig_path(path).write_text(sig + "\n", encoding="utf-8")
    return sig


def verify_file(path: Path, key: bytes) -> None:
    """None, если подпись верна; иначе ReplayTamperedError с понятным текстом."""
    path = Path(path)
    sp = _sig_path(path)
    if not sp.exists():
        raise ReplayTamperedError(f"у {path.name} нет файла подписи {sp.name}: записанный прогон не принят. "
                                  f"Если файл ваш, подпишите его: python -m app.security sign {path}")
    want = hmac.new(key, path.read_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, sp.read_text(encoding="utf-8").strip()):
        raise ReplayTamperedError(f"подпись {path.name} не сходится: файл изменён после записи или подписан другим ключом "
                                  f"(RADAR_HMAC_KEY). Записанный прогон не принят. Пересоберите: make demo-record")


def load_recorded(path: Path, key_file: Path) -> dict:
    """Читает записанные ответы агента, проверив подпись. Нет файла: пустой набор (как раньше).
    Первый запуск (ключ только что создан и подписи ещё нет): файл подписывается как доверенный (trust on first use)."""
    import json
    path = Path(path)
    if not path.exists():
        return {"results": {}}
    key, created = get_key(key_file)
    if not _sig_path(path).exists() and created:
        sign_file(path, key)
    verify_file(path, key)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- токены и роли
VIEWER, DUTY = "viewer", "duty"


def token_roles() -> dict:
    """{токен: роль} из окружения на этот момент. RADAR_API_TOKEN и RADAR_TOKEN_DUTY дают роль «дежурный» (кнопки решений),
    RADAR_TOKEN_VIEW только чтение. Пусто: вход выключен (демо на 127.0.0.1)."""
    out = {}
    for var, role in (("RADAR_TOKEN_VIEW", VIEWER), ("RADAR_API_TOKEN", DUTY), ("RADAR_TOKEN_DUTY", DUTY)):
        v = os.environ.get(var)
        if v:
            out[v] = role
    return out


def role_of(auth_header: str | None, roles: dict) -> str | None:
    """Роль по заголовку Authorization: Bearer <токен>; None, если токена нет или он неверный. Сравнение за постоянное время."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    given = auth_header[7:].strip().encode("utf-8")
    found = None
    for tok, role in roles.items():
        if hmac.compare_digest(given, tok.encode("utf-8")):
            found = role
    return found


# ---------------------------------------------------------------- лимит частоты (скользящее окно в памяти)
class SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_s: float, clock=time.monotonic):
        self.max_calls, self.window_s, self.clock = max_calls, window_s, clock
        self._hits: dict = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.max_calls:
                return False
            q.append(now)
            if len(self._hits) > 10_000:   # память ограничена: чужие давно пустые ключи выбрасываем
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] >= self.window_s]:
                    del self._hits[k]
            return True

    def reset(self):
        with self._lock:
            self._hits.clear()


# ---------------------------------------------------------------- token bucket (запас запросов + равномерное пополнение)
class TokenBucketLimiter:
    """Ведро токенов на клиента: вмещает burst запросов сразу, пополняется rate_per_s токенов в секунду. В отличие от скользящего окна
    разрешает короткий всплеск (человек быстро нажал несколько кнопок), но не даёт держать высокую частоту подолгу.
    allow(key) -> (разрешено, через сколько секунд появится токен). Память ограничена max_keys клиентами."""

    def __init__(self, rate_per_s: float, burst: int, clock=time.monotonic, max_keys: int = 10_000):
        self.rate, self.burst, self.clock, self.max_keys = rate_per_s, burst, clock, max_keys
        self._state: dict = {}          # key -> [токены, время последнего пересчёта]
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple:
        now = self.clock()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = [tokens - 1.0, now]
                ok, retry = True, 0.0
            else:
                self._state[key] = [tokens, now]
                ok, retry = False, (1.0 - tokens) / self.rate if self.rate > 0 else float("inf")
            if len(self._state) > self.max_keys:   # ведро, которое успело наполниться, равно новому: забываем его
                full = [k for k, (t, l) in self._state.items() if min(float(self.burst), t + (now - l) * self.rate) >= self.burst]
                for k in full:
                    del self._state[k]
            return ok, retry

    def reset(self):
        with self._lock:
            self._state.clear()


# ---------------------------------------------------------------- бюджет вызовов LLM
class LlmBudget:
    """Не больше per_product вызовов модели в час на продукт и total в час в целом (скользящий час). Агент вызывается на карточку
    (всплеск), а не на обращение; бюджет страхует от потока всплесков (накрутка по многим продуктам, сбой порога).
    try_acquire(product) -> (можно, причина отказа). Исчерпан: карточка собирается шаблоном без модели, это пишется в аудит."""

    def __init__(self, per_product: int, total: int, clock=time.monotonic, window_s: float = 3600.0):
        self.per_product, self.total, self.clock, self.window_s = per_product, total, clock, window_s
        self._by: dict = {}
        self._all: deque = deque()
        self._lock = threading.Lock()

    def _trim(self, q: deque, now: float):
        while q and now - q[0] >= self.window_s:
            q.popleft()

    def try_acquire(self, product: str) -> tuple:
        now = self.clock()
        with self._lock:
            q = self._by.setdefault(product, deque())
            self._trim(q, now)
            self._trim(self._all, now)
            if len(q) >= self.per_product:
                return False, f"лимит вызовов модели на продукт {product}: {self.per_product} в час"
            if len(self._all) >= self.total:
                return False, f"общий лимит вызовов модели: {self.total} в час"
            q.append(now)
            self._all.append(now)
            return True, ""

    def used(self) -> dict:
        now = self.clock()
        with self._lock:
            self._trim(self._all, now)
            return {"total": len(self._all), "limit_total": self.total, "limit_per_product": self.per_product}


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "sign":
        k, _ = get_key(Path(__file__).resolve().parents[1] / "demo" / ".replay-hmac.key")
        print("подписано:", sign_file(Path(sys.argv[2]), k))
    else:
        print("использование: python -m app.security sign demo/replay.json")
