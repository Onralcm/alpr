import time

class Timer:
    def __init__(self):
        self._t = {}
        self._start = time.perf_counter()

    def mark(self, name: str):
        self._t[name] = time.perf_counter()

    def ms(self, a: str, b: str) -> float:
        return (self._t[b] - self._t[a]) * 1000.0

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
