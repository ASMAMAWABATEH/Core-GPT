from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Logger:
    """Simple stdout logger with timestamps."""

    logfile: Optional[str] = None

    def _write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        if self.logfile:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def info(self, message: str) -> None:
        self._write(message)

    def warn(self, message: str) -> None:
        self._write(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self._write(f"ERROR: {message}")
