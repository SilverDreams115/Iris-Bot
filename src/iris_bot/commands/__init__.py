from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from iris_bot.config import Settings


CommandHandler = Callable[[Settings], int]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    handler: CommandHandler
    domain: str
