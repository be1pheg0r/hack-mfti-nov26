"""Immutable, serializable contracts for repeatable benchmark runs."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkContract(BaseModel):
    """Base for immutable benchmark data with strict field validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BenchmarkSpec(BenchmarkContract):
    """Configuration for one named callable benchmark."""

    name: str = Field(min_length=1)
    iterations: int = Field(gt=0)
    warmup_iterations: int = Field(default=0, ge=0)


class BenchmarkSample(BenchmarkContract):
    """One measured invocation; errors are retained rather than hidden."""

    iteration: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    success: bool
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> "BenchmarkSample":
        """Require errors only on failed invocations."""
        if self.success == (self.error_type is not None):
            raise ValueError("Successful samples have no error; failures require one")
        return self


class BenchmarkResult(BenchmarkContract):
    """Measured samples and computed aggregate duration statistics."""

    name: str = Field(min_length=1)
    samples: tuple[BenchmarkSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_iterations(self) -> "BenchmarkResult":
        """Require unique, ordered, one-based measured iteration numbers."""
        if tuple(s.iteration for s in self.samples) != tuple(
            range(1, len(self.samples) + 1)
        ):
            raise ValueError("Sample iterations must be consecutive and one-based")
        return self

    @property
    def successful_iterations(self) -> int:
        """Number of measured invocations that completed successfully."""
        return sum(sample.success for sample in self.samples)

    @property
    def failed_iterations(self) -> int:
        """Number of measured invocations that raised an exception."""
        return len(self.samples) - self.successful_iterations

    @property
    def mean_seconds(self) -> float | None:
        """Arithmetic mean of successful durations, or None if all failed."""
        durations = self._successful_durations()
        return sum(durations) / len(durations) if durations else None

    @property
    def median_seconds(self) -> float | None:
        """Median of successful durations, or None if all failed."""
        durations = sorted(self._successful_durations())
        if not durations:
            return None
        middle = len(durations) // 2
        if len(durations) % 2:
            return durations[middle]
        return (durations[middle - 1] + durations[middle]) / 2

    @property
    def p95_seconds(self) -> float | None:
        """Nearest-rank 95th percentile of successful durations."""
        durations = sorted(self._successful_durations())
        if not durations:
            return None
        index = (95 * len(durations) + 99) // 100 - 1
        return durations[index]

    def _successful_durations(self) -> list[float]:
        """Return durations from successful samples only."""
        return [sample.duration_seconds for sample in self.samples if sample.success]
