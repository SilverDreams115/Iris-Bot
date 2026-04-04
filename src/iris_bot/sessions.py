from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


SESSION_ORDER = ("asia", "london", "new_york", "off_session")


@dataclass(frozen=True)
class SessionDefinition:
    name: str
    start_hour: int
    end_hour: int

    def contains(self, hour: int) -> bool:
        return self.start_hour <= hour < self.end_hour


SESSION_DEFINITIONS = (
    SessionDefinition("asia", 0, 8),
    SessionDefinition("london", 8, 13),
    SessionDefinition("new_york", 13, 21),
)


def canonical_session_name(timestamp: datetime) -> str:
    hour = timestamp.hour
    for session in SESSION_DEFINITIONS:
        if session.contains(hour):
            return session.name
    return "off_session"


def session_flags(timestamp: datetime) -> tuple[float, float, float]:
    current = canonical_session_name(timestamp)
    return (
        1.0 if current == "asia" else 0.0,
        1.0 if current == "london" else 0.0,
        1.0 if current == "new_york" else 0.0,
    )


def session_definition_report() -> dict[str, object]:
    return {
        "canonical_order": list(SESSION_ORDER),
        "definitions": [asdict(item) for item in SESSION_DEFINITIONS],
        "off_session": {"name": "off_session", "hours": "21:00-23:59"},
    }
