from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type] = {}

    def register(self, key: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            self._items[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"{self.name} registry missing key: {key}") from exc

    def build(self, config: dict[str, Any], **extra: Any) -> Any:
        params = dict(config)
        kind = params.pop("type")
        cls = self.get(kind)
        params.update(extra)
        return cls(**params)
