"""Pydantic request models for the analysis API."""

from pydantic import BaseModel, Field

from src.backend.settings import AGILE_MODELING_PARAMS, EVALUATION_PARAMS, DEFAULT_SAMPLE_SIZE


class SelectRequest(BaseModel):
    species: str


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default_factory=lambda: AGILE_MODELING_PARAMS["initialisation_nearest_neighbours"])
    n_dissimilar: int = Field(default_factory=lambda: AGILE_MODELING_PARAMS["initialisation_n_dissimilar"])


class AnnotateRequest(BaseModel):
    labels: dict[str, int | None]


class SampleRequest(BaseModel):
    target_logit: float = 0.0
    n: int = DEFAULT_SAMPLE_SIZE


class EvalStartRequest(BaseModel):
    n_buckets: int = EVALUATION_PARAMS["n_buckets"]
    k: int = EVALUATION_PARAMS["labels_per_bucket"]
