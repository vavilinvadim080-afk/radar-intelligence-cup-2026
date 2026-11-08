"""Проходит четыре сцены демо автоматически (для записи видео) и печатает время каждой.

Сервер должен быть запущен (`make demo`). Страница открыта в браузере: скрипт жмёт те же кнопки через HTTP,
страница обновляется сама. Пример: python demo/run_scenes.py --base http://127.0.0.1:8777 --hold 4
"""
from __future__ import annotations

import argparse
import time

import httpx


class Scenes:
    def __init__(self, base: str, hold: float, timeout: float):
        self.c = httpx.Client(base_url=base, timeout=60)
        self.hold, self.timeout = hold, timeout
        self.times: list = []

    def post(self, path, **body):
        r = self.c.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def get(self, path):
        return self.c.get(path).json()

    def wait_card(self, card_id: str):
        """Ждём, пока карточка появилась и агент закончил (pending = false)."""
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            for c in self.get("/api/cards")["cards"]:
                if c["id"] == card_id and not c["pending"]:
                    return c
            time.sleep(0.5)
        raise TimeoutError(f"карточка {card_id} не появилась")

    def wait_fix(self, card_id: str):
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            for c in self.get("/api/cards")["cards"]:
                if c["id"] == card_id and c["close_proposal"]:
                    return c
            time.sleep(0.5)
        raise TimeoutError("проверка фикса не пришла")

    def play(self, speed: float):
        self.post("/api/control", speed=speed, playing=True)

    def pause(self):
        self.post("/api/control", playing=False)

    def pause_for(self, k: float = 1.0):
        time.sleep(self.hold * k)

    def scene(self, name, fn):
        t = time.time()
        print(f"--- {name}", flush=True)
        fn()
        dt = time.time() - t
        self.times.append((name, dt))
        print(f"    {name}: {dt:.0f} с", flush=True)

    # ------------------------------------------------------------ сцены
    def s1(self):
        self.post("/api/jump", scene=1, reset=True)
        self.play(5)
        self.pause_for(3)            # поток идёт, маскирование видно в нижней панели
        self.pause()

    def s2(self):
        self.post("/api/jump", scene=2, reset=True)
        self.play(20)                # релиз 2.8.0 в 14:45, карточка около 18:00
        self.wait_card("B0003")
        self.pause()
        self.pause_for(2)

    def s3(self):
        self.post("/api/cards/B0003/decision", action="our_bug")      # «наш баг» -> задача в мок-трекере
        self.pause_for(1.5)
        self.post("/api/jump", to="2026-03-11T20:40")
        self.play(10)
        c = self.wait_card("B0005")                                    # ложная тревога по азс
        self.pause()
        self.pause_for(1)
        self.post("/api/cards/B0005/decision", action="noise")        # «шум» -> порог продукта вверх
        self.pause_for(1.5)
        self.post("/api/jump", to="2026-03-12T16:00")
        self.play(10)
        c = self.wait_card("B0006")
        self.pause()
        key = c["agent"]["card"].get("existing_ticket")
        if not key:                                                    # агент не нашёл задачу: берём первую «в работе» по продукту
            key = next(t["key"] for t in self.get("/api/tracker")["tasks"] if t["product"] == "portal" and t["status"] == "in_progress")
        self.pause_for(1)
        self.post("/api/cards/B0006/decision", action="known", task_key=key)
        self.pause_for(1.5)

    def s4(self):
        self.play(30)                                                  # фикс 2.8.1 в 13:15 уже прошёл, тема гаснет
        self.pause_for(4)
        self.pause()
        self.post("/api/jump", to="2026-03-14T05:50")
        self.play(10)
        self.wait_card("W-T012906")                                    # обращение с инструкцией: агент помечает и ничего не делает
        self.pause()
        self.pause_for(3)
        self.post("/api/jump", to="2026-03-16T22:30")
        self.play(10)
        self.wait_fix("B0003")
        self.pause()
        self.pause_for(2)
        self.post("/api/cards/B0003/confirm-close")                    # закрывает человек
        self.pause_for(2)

    def run(self):
        for name, fn in (("1 Поток и маскирование", self.s1), ("2 Релиз и всплеск", self.s2),
                         ("3 Человек в контуре", self.s3), ("4 Петля и инъекция", self.s4)):
            self.scene(name, fn)
        print("итого: %.0f с" % sum(t for _, t in self.times))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8777")
    ap.add_argument("--hold", type=float, default=4.0, help="базовая пауза на кадр, секунд")
    ap.add_argument("--timeout", type=float, default=900, help="сколько ждать карточку (в живом режиме LLM медленный)")
    a = ap.parse_args()
    Scenes(a.base, a.hold, a.timeout).run()
