"""Benchmark result dataclasses with JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PowerResult:
    avg_watts: float = 0.0
    peak_watts: float = 0.0
    samples: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    sampling_interval_ms: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PowerResult:
        return cls(**d)


@dataclass
class BenchmarkResult:
    system: str
    benchmark_name: str
    results: list[dict[str, Any]]
    power: PowerResult
    device_info: dict[str, Any]
    timestamp: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> BenchmarkResult:
        with open(path) as f:
            d = json.load(f)
        d["power"] = PowerResult.from_dict(d["power"])
        return cls(**d)
