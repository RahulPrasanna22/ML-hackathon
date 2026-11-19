"""Pydantic schemas implementing docs/API_CONTRACT.md (the Laptop 2 contract).

Conventions (from the contract): coordinates are [lon, lat]; fractions in [0, 1];
times in milliseconds with keys ending in `_ms`; trip ids are zero-padded strings;
errors use FastAPI's default {"detail": ...} shape.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description='"ok" when the model is loaded, otherwise "degraded"')
    model_loaded: bool
    model_name: Optional[str] = None
    hmm_enabled: bool
    version: str
    num_evaluation_trips: int
    # Laptop-1 extras (ignored by the UI)
    evaluation_partition: str
    evaluation_partition_is_fallback: bool
    uptime_s: float


class TripSummary(BaseModel):
    trip_id: str
    num_observations: Optional[int] = Field(None, description="null if the trip's .track file is unreadable")
    duration_s: Optional[float] = Field(None, description="null if the trip's .track file is unreadable")
    label: Optional[str] = None


class TripListResponse(BaseModel):
    partition: str
    trips: list[TripSummary]


class RunRequest(BaseModel):
    trip_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[0-9A-Za-z_\-]+$", examples=["00000007"])


class Observation(BaseModel):
    index: int = Field(description="Row index in the raw .track file")
    lon: float
    lat: float
    t: float
    matched_arc_id: Optional[int] = Field(None, description="PREDICTED arc (HMM output); null if unmatched")


class Segment(BaseModel):
    arc_id: int
    coordinates: list[list[float]] = Field(description="[[lon, lat], [lon, lat]] (from-node, to-node)")


class RouteGeometry(BaseModel):
    arc_ids: list[int] = []
    segments: list[Segment] = []


class HmmInfo(BaseModel):
    applied: bool
    status: str
    message: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    version: str
    algorithm: Optional[str] = None
    family: Optional[str] = None
    created_utc: Optional[str] = None
    selection_metric: Optional[str] = None


class EvaluationResult(BaseModel):
    evaluation_id: str
    trip_id: str
    status: Literal["completed", "failed"]
    created_at: str
    error: Optional[str] = None
    partition: str
    model: ModelInfo
    num_observations: int
    observations: list[Observation]
    predicted_route: RouteGeometry
    classifier_only_route: Optional[RouteGeometry] = None
    road_network: RouteGeometry
    metrics: dict[str, Optional[float]] = Field(description="Classifier + HMM/Viterbi, scored server-side; {} if no reference")
    classifier_only_metrics: Optional[dict[str, Optional[float]]] = None
    nearest_road_metrics: Optional[dict[str, Optional[float]]] = None
    timing: dict[str, float]
    hmm: HmmInfo
    hmm_params: dict
    folium_map_url: Optional[str] = None
    warnings: list[str] = []
