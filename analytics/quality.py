"""Data-quality tracking.

Each analysis records what it *included*, what it *excluded* and what merely
*reduces confidence*, so the narration layer can attach an honest caveat instead
of presenting an unqualified number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Severity", "QualityIssue", "DataQualityReport"]


class Severity(str, Enum):
    #: Rows were dropped from the calculation.
    EXCLUDED = "excluded"
    #: Rows were included but a field was imputed/unknown.
    INCLUDED_WITH_GAP = "included_with_gap"
    #: Background note that may reduce confidence.
    INFO = "info"


@dataclass(frozen=True)
class QualityIssue:
    """One data problem, always quantified."""

    code: str
    message: str
    count: int
    severity: Severity = Severity.INFO
    total: int | None = None
    field: str | None = None

    @property
    def share(self) -> float | None:
        if not self.total:
            return None
        return self.count / self.total

    def as_dict(self) -> dict:
        payload = {
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "severity": self.severity.value,
        }
        if self.total:
            payload["of_total"] = self.total
            payload["share_pct"] = round(100 * self.count / self.total, 1)
        if self.field:
            payload["field"] = self.field
        return payload


@dataclass
class DataQualityReport:
    """Collection of :class:`QualityIssue` for one dataset or analysis."""

    dataset: str
    row_count: int = 0
    issues: list[QualityIssue] = field(default_factory=list)

    def add(
        self,
        code: str,
        message: str,
        count: int,
        severity: Severity = Severity.INFO,
        *,
        total: int | None = None,
        field_name: str | None = None,
    ) -> None:
        """Record an issue. Zero-count issues are ignored to avoid noise."""
        if count <= 0:
            return
        self.issues.append(
            QualityIssue(
                code=code,
                message=message,
                count=int(count),
                severity=severity,
                total=total if total is not None else (self.row_count or None),
                field=field_name,
            )
        )

    def extend(self, other: "DataQualityReport | None") -> None:
        if other:
            self.issues.extend(other.issues)

    def by_severity(self, severity: Severity) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == severity]

    def top(self, limit: int = 5) -> list[QualityIssue]:
        """Most material issues first: excluded rows outrank informational notes."""
        order = {Severity.EXCLUDED: 0, Severity.INCLUDED_WITH_GAP: 1, Severity.INFO: 2}
        return sorted(self.issues, key=lambda i: (order[i.severity], -i.count))[:limit]

    def as_dict(self, limit: int | None = None) -> dict:
        issues = self.top(limit) if limit else self.issues
        return {
            "dataset": self.dataset,
            "rows": self.row_count,
            "issues": [i.as_dict() for i in issues],
        }

    def __bool__(self) -> bool:
        return bool(self.issues)
