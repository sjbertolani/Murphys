from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OptionRight(StrEnum):
    CALL = "C"
    PUT = "P"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UnderlyingBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    adjusted_close: float | None = None


class OptionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    option_symbol: str
    quote_timestamp: datetime
    expiration: datetime
    strike: float
    right: OptionRight = OptionRight.CALL
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    spot: float | None = None

    @field_validator("strike")
    @classmethod
    def strike_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("strike must be positive")
        return value


class OptionExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    symbol: str
    option_symbol: str
    forecast_timestamp: datetime
    expiration: datetime
    strike: float
    spot: float
    dte: float
    moneyness: float
    label: int | None = Field(default=None, ge=0, le=1)
    resolution_timestamp: datetime | None = None


class BeliefState(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    confidence: Confidence = Confidence.LOW
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    update_reasoning: str = ""


class Forecast(BaseModel):
    example_id: str
    method: str
    raw_probability: float = Field(ge=0.0, le=1.0)
    aggregate_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime

