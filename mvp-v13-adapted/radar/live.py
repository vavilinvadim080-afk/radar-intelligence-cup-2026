"""Потоковый приём обращений с контрольными точками: то, чего нет в пакетном run_full.

LiveRadar принимает обращения по одному (ingest). До конца прогрева (warmup_days) копит их в списке: эмбеддер обучается
на прогревочных текстах, как в пакетном _prepare. Дальше каждое обращение: маскирование -> эмбеддинг -> RadarStream.push.

Контрольная точка (save_checkpoint / load_checkpoint) хранит ЖУРНАЛ принятых обращений, а не внутренности детектора:
  * id, время, продукт, канал, платформа, версия, user_id, ЗАМАСКИРОВАННЫЙ текст, вектор (float32, base64);
  * версию формата, релизы, Config, число обращений прогрева.
Восстановление заново проигрывает журнал через тот же RadarStream.push. Детектор детерминирован, поэтому состояние (кластеры,
EWMA-базовая линия, открытые всплески, сезонность, дедуп голосов) получается тем же, что до перезапуска; это проверяет
tests/test_live.py. Цена: время восстановления пропорционально числу обращений (порядка 1/6000 с на обращение), см. REPORT-load.md.
Исходный текст с ПДн в файл не пишется.

Запись атомарная: временный файл в той же папке, fsync, os.replace. Повреждённый, пустой или чужой версии файл не роняет
систему: load_checkpoint возвращает чистый старт и пишет событие checkpoint_rejected в журнал.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np

from .config import Config
from .embed import make_embedder
from .mask import mask_text
from .models import Release, Ticket
from .pipeline import RadarStream

FORMAT_VERSION = 1


def _noop(kind: str, payload: dict) -> None:
    return None


class LiveRadar:
    def __init__(self, releases: list[Release], cfg: Optional[Config] = None, log: Callable[[str, dict], None] = _noop):
        self.cfg = cfg or Config()
        self.releases = releases
        self.log = log                                  # log(kind, payload): сюда идут отказы, повторы, восстановление
        self.emb = None
        self.stream: Optional[RadarStream] = None
        self.buf: list[tuple] = []                      # прогрев: (Ticket, замаскированный текст)
        self.warm_n = 0                                 # сколько обращений ушло на обучение эмбеддера
        self.start_ts: Optional[datetime] = None
        self._buf_ids: set = set()
        self._buf_last: Optional[datetime] = None

    # ------------------------------------------------------------ приём
    def ingest(self, tk: Ticket) -> str:
        """Принимает одно обращение. Возвращает вердикт: ok / duplicate / late_clamped / too_late."""
        if self.stream is None:
            return self._ingest_warm(tk)
        masked = mask_text(tk.text, ner=self.cfg.mask_ner, ctx_names=self.cfg.mask_ctx_names, cascade_cfg=self.cfg)[0]
        verdict = self.stream.admit(tk)[0]
        if verdict in ("duplicate", "too_late", "user_capped"):   # не тратим эмбеддинг на то, что не примем
            self.stream.push(tk, masked, np.zeros(self.stream.dim, dtype=np.float32))
            self._log_verdict(tk, verdict)
            return verdict
        vec = self.emb.transform([masked])[0]
        verdict = self.stream.push(tk, masked, vec)
        self._log_verdict(tk, verdict)
        return verdict

    def ingest_many(self, tickets, chunk: int = 256) -> dict:
        """Пачка обращений. После прогрева эмбеддинг считается пачками по chunk штук (быстрее в ~4 раза, чем по одному);
        проверки повтора и опоздания идут по каждому обращению по очереди, как в ingest."""
        out: dict = {}
        tickets = list(tickets)
        i = 0
        while i < len(tickets):
            if self.stream is None:
                v = self.ingest(tickets[i])
                out[v] = out.get(v, 0) + 1
                i += 1
                continue
            part = tickets[i:i + chunk]
            masked = [mask_text(t.text, ner=self.cfg.mask_ner, ctx_names=self.cfg.mask_ctx_names, cascade_cfg=self.cfg)[0] for t in part]
            vecs = self.emb.transform(masked)
            for tk, m, v in zip(part, masked, vecs):
                verdict = self.stream.push(tk, m, v)
                self._log_verdict(tk, verdict)
                out[verdict] = out.get(verdict, 0) + 1
            i += chunk
        return out

    def _log_verdict(self, tk: Ticket, verdict: str) -> None:
        if verdict not in ("ok", "user_capped"):   # user_capped пишет сам поток (antispam_user_capped, не чаще чем каждое сотое): иначе флуд забьёт журнал
            self.log("ticket_" + verdict, {"ticket_id": tk.id, "ts": tk.ts.isoformat(timespec="seconds")})

    def _ingest_warm(self, tk: Ticket) -> str:
        """Прогрев: копим обращения. Те же правила повтора и опоздания, что в RadarStream.admit."""
        if tk.id in self._buf_ids:
            self._log_verdict(tk, "duplicate")
            return "duplicate"
        verdict = "ok"
        if self._buf_last is not None and tk.ts < self._buf_last:
            if (self._buf_last - tk.ts) > timedelta(minutes=self.cfg.late_tolerance_min):
                self._log_verdict(tk, "too_late")
                return "too_late"
            tk, verdict = replace(tk, ts=self._buf_last), "late_clamped"
            self._log_verdict(tk, verdict)
        if self.start_ts is None:
            self.start_ts = tk.ts
        if tk.ts >= self.start_ts + timedelta(days=self.cfg.warmup_days):
            self._start_stream()
            self._buf_ids, self._buf_last = set(), None
            return self.ingest(tk)
        self._buf_ids.add(tk.id)
        self._buf_last = tk.ts
        masked = mask_text(tk.text, ner=self.cfg.mask_ner, ctx_names=self.cfg.mask_ctx_names, cascade_cfg=self.cfg)[0]
        self.buf.append((tk, masked))
        return verdict

    def _start_stream(self, vecs: Optional[np.ndarray] = None) -> None:
        """Конец прогрева: учим эмбеддер на накопленных текстах, создаём поток, проигрываем накопленное."""
        cfg = self.cfg
        self.warm_n = len(self.buf)
        self.emb = make_embedder(cfg.embedder, cfg.svd_dim)
        self.emb.fit([m for _, m in self.buf])
        X = vecs if vecs is not None else self.emb.transform([m for _, m in self.buf])
        start = self.start_ts
        t0 = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        self.stream = RadarStream(self.releases, cfg, X.shape[1], t0, start)
        self.stream.embedder_name = self.emb.name
        self.stream.on_antispam = self.log
        for (tk, masked), v in zip(self.buf, X):
            self.stream.push(tk, masked, v)
        self.buf = []

    # ------------------------------------------------------------ результат
    @property
    def bursts(self) -> list:
        return self.stream.bursts if self.stream else []

    def n_accepted(self) -> int:
        return len(self.stream.tickets) if self.stream else len(self.buf)

    # ------------------------------------------------------------ контрольная точка
    def _accepted(self) -> list[tuple]:
        """[(Ticket, masked, vec|None)] в порядке приёма."""
        if self.stream is None:
            return [(tk, m, None) for tk, m in self.buf]
        st = self.stream
        return [(tk, st.texts[i], st.vecs[i]) for i, tk in enumerate(st.tickets)]

    def save_checkpoint(self, path: str) -> dict:
        rows = []
        for tk, masked, vec in self._accepted():
            rows.append({"id": tk.id, "ts": tk.ts.isoformat(), "product": tk.product, "channel": tk.channel, "platform": tk.platform,
                         "app_version": tk.app_version, "user_id": tk.user_id, "text": masked})
        vecs = None
        if self.stream is not None:
            arr = np.ascontiguousarray(self.stream.vecs[:len(self.stream.tickets)], dtype="<f4")
            vecs = base64.b64encode(arr.tobytes()).decode("ascii")
        body = {
            "format_version": FORMAT_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "warm_n": self.warm_n, "dim": int(self.stream.dim) if self.stream else None,
            "config": self.cfg.model_dump(),
            "releases": [{"product": r.product, "version": r.version, "ts": r.ts.isoformat(), "kind": r.kind, "changelog": r.changelog}
                         for r in self.releases],
            "tickets": rows, "vecs_f32_b64": vecs,
            "summary": self.summary(),
        }
        body["checksum"] = _checksum(body)
        atomic_write_json(path, body)
        self.log("checkpoint_saved", {"path": path, "tickets": len(rows), "bursts": len(self.bursts)})
        return body["summary"]

    def summary(self) -> dict:
        return {"tickets": self.n_accepted(), "bursts": [[b.id, b.product, b.detected_at.isoformat(), b.alert] for b in self.bursts]}

    @classmethod
    def load_checkpoint(cls, path: str, releases: list[Release], cfg: Optional[Config] = None,
                        log: Callable[[str, dict], None] = _noop) -> "LiveRadar":
        """Восстанавливает LiveRadar из файла. При любой беде (нет файла, пусто, битый JSON, чужая версия, не сошлась
        контрольная сумма, сломанные данные) возвращает чистый LiveRadar и пишет checkpoint_rejected."""
        cfg = cfg or Config()
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            if not raw.strip():
                raise ValueError("файл пуст")
            body = json.loads(raw)
            if not isinstance(body, dict) or body.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"версия формата {body.get('format_version') if isinstance(body, dict) else '?'}, ожидалась {FORMAT_VERSION}")
            if body.get("checksum") != _checksum(body):
                raise ValueError("контрольная сумма не сошлась")
            live = cls(releases, cfg, log)
            live._restore(body)
        except Exception as e:   # noqa: BLE001  любая беда с файлом = чистый старт, система не падает
            log("checkpoint_rejected", {"path": path, "reason": f"{type(e).__name__}: {e}"})
            return cls(releases, cfg, log)
        saved_cfg = body.get("config", {})
        if saved_cfg != cfg.model_dump():
            changed = sorted(k for k in set(saved_cfg) | set(cfg.model_dump()) if saved_cfg.get(k) != cfg.model_dump().get(k))
            live.log("checkpoint_config_changed", {"fields": changed, "note": "состояние пересчитано с текущим Config"})
        if live.summary() != body.get("summary"):
            live.log("checkpoint_summary_differs", {"saved": body.get("summary"), "restored": live.summary()})
        live.log("checkpoint_loaded", {"path": path, "tickets": live.n_accepted(), "bursts": len(live.bursts)})
        return live

    def _restore(self, body: dict) -> None:
        tickets = [(Ticket(id=r["id"], ts=datetime.fromisoformat(r["ts"]), product=r["product"], channel=r["channel"],
                           platform=r["platform"], app_version=r["app_version"], user_id=r["user_id"], text=r["text"]), r["text"])
                   for r in body["tickets"]]
        if not tickets:
            return
        self.start_ts = tickets[0][0].ts
        if body.get("vecs_f32_b64") is None:   # сохранили во время прогрева
            self.buf = tickets
            self._buf_ids = {t.id for t, _ in tickets}
            self._buf_last = tickets[-1][0].ts
            return
        dim = body["dim"]
        X = np.frombuffer(base64.b64decode(body["vecs_f32_b64"]), dtype="<f4").reshape(-1, dim)
        if len(X) != len(tickets):
            raise ValueError("число векторов не равно числу обращений")
        warm_n = body["warm_n"]
        # эмбеддер учим на тех же прогревочных текстах, что и до перезапуска (детерминированно: random_state=0)
        self.buf = tickets[:warm_n]
        self._start_stream(vecs=X[:warm_n].astype(np.float64))
        self._buf_ids, self._buf_last = set(), None
        for (tk, masked), v in zip(tickets[warm_n:], X[warm_n:]):
            self.stream.push(tk, masked, v)


def _checksum(body: dict) -> str:
    core = {k: v for k, v in body.items() if k != "checksum"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def atomic_write_json(path: str, obj) -> None:
    """Пишет JSON во временный файл рядом и переименовывает: читатель видит либо старую, либо новую версию целиком."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
