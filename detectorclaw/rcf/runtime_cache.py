from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Generic
from typing import TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    def __init__(self, max_entries: int) -> None:
        self._max_entries = max(1, int(max_entries))
        self._items: OrderedDict[object, T] = OrderedDict()
        self._lock = Lock()

    def get(self, key: object) -> T | None:
        with self._lock:
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def set(self, key: object, value: T) -> T:
        with self._lock:
            if key in self._items:
                self._items.pop(key)
            self._items[key] = value
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def items(self) -> list[tuple[object, T]]:
        with self._lock:
            return list(self._items.items())


def file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
