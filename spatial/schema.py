"""Inference-only contracts. No gold fields belong in these objects."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Literal

OPERATIONS = (
    "nearest_category", "cardinal_direction", "within_radius_yes_no",
    "closer_of_two", "count_within_radius", "nearest_of_two_categories",
    "two_hop_nearest", "spatial_multi_constraint",
)
CATEGORIES = frozenset(("مستشفى", "عيادة", "صيدلية", "مدرسة", "جامعة", "كلية",
    "مطعم", "مقهى", "بنك", "محطة وقود", "مطعم وجبات سريعة", "صراف آلي",
    "روضة أطفال", "مكان عبادة"))
DIRECTIONS = ("شمال", "شرق", "جنوب", "غرب")
Status = Literal["success", "ambiguous", "not_found", "unsupported", "invalid_query"]


@dataclass(frozen=True)
class Location:
    identity: str  # context-scoped ordinal; never an OSM/gold ID
    name: str
    latitude: float
    longitude: float
    category: str | None = None

    @property
    def coordinates(self):
        return self.latitude, self.longitude


@dataclass(frozen=True)
class ParsedContext:
    anchor: Location | None
    candidates: tuple[Location, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityReference:
    name: str
    source: Literal["context_anchor", "candidate", "unspecified"] = "unspecified"


@dataclass(frozen=True)
class StructuredQuery:
    operation: str
    origin: EntityReference
    targets: tuple[EntityReference, ...] = ()
    categories: tuple[str, ...] = ()
    radius_km: float | None = None
    direction: str | None = None
    first_hop_category: str | None = None
    second_hop_category: str | None = None


@dataclass(frozen=True)
class Resolution:
    reference: EntityReference
    status: Literal["UNIQUE", "AMBIGUOUS", "NOT_FOUND"]
    matches: tuple[Location, ...]
    message: str

    @property
    def candidate_count(self):
        return len(self.matches)


@dataclass(frozen=True)
class TypedAnswer:
    kind: Literal["entity", "boolean", "integer", "direction"]
    value: str | bool | int
    text: str


@dataclass
class ExecutionResult:
    status: Status
    operation: str | None
    answer: TypedAnswer | None = None
    reason: str | None = None
    stage: str = "executor"
    trace: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class QueryError(ValueError):
    def __init__(self, status: Status, reason: str, stage="parser"):
        super().__init__(reason)
        self.status, self.reason, self.stage = status, reason, stage
