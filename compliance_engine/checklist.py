"""
Automated Checklist Generation
================================
Generates compliance checklists from AIS-175 clauses.
Each checklist item maps to a clause requirement and tracks completion status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from .clauses import (
    AIS175_CLAUSES,
    Clause,
    ClauseCategory,
    Requirement,
    Severity,
    get_all_clauses,
    get_clauses_by_category,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckItemStatus(str, Enum):
    PENDING = "PENDING"
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    WAIVED = "WAIVED"


@dataclass
class ChecklistItem:
    """A single item on the compliance checklist."""

    item_id: str
    clause_id: str
    req_id: str
    title: str
    description: str
    mandatory: bool
    severity: str
    category: str
    status: CheckItemStatus = CheckItemStatus.PENDING
    notes: str = ""
    reviewed_by: str = ""
    reviewed_at: Optional[str] = None

    def mark_compliant(self, reviewer: str = "", notes: str = "") -> None:
        self.status = CheckItemStatus.COMPLIANT
        self.reviewed_by = reviewer
        self.reviewed_at = _now()
        if notes:
            self.notes = notes

    def mark_non_compliant(self, reviewer: str = "", notes: str = "") -> None:
        self.status = CheckItemStatus.NON_COMPLIANT
        self.reviewed_by = reviewer
        self.reviewed_at = _now()
        if notes:
            self.notes = notes

    def mark_not_applicable(self, reason: str = "") -> None:
        self.status = CheckItemStatus.NOT_APPLICABLE
        self.reviewed_at = _now()
        if reason:
            self.notes = reason

    def waive(self, reason: str, reviewer: str) -> None:
        self.status = CheckItemStatus.WAIVED
        self.reviewed_by = reviewer
        self.reviewed_at = _now()
        self.notes = f"WAIVED: {reason}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Checklist:
    """Complete AIS-175 compliance checklist for a vehicle / ECU submission."""

    checklist_id: str
    vehicle_model: str
    manufacturer: str
    generated_at: str = field(default_factory=_now)
    items: List[ChecklistItem] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def compliant_count(self) -> int:
        return sum(1 for i in self.items if i.status == CheckItemStatus.COMPLIANT)

    @property
    def non_compliant_count(self) -> int:
        return sum(1 for i in self.items if i.status == CheckItemStatus.NON_COMPLIANT)

    @property
    def pending_count(self) -> int:
        return sum(1 for i in self.items if i.status == CheckItemStatus.PENDING)

    @property
    def critical_non_compliant(self) -> List[ChecklistItem]:
        return [
            i for i in self.items
            if i.status == CheckItemStatus.NON_COMPLIANT
            and i.severity == Severity.CRITICAL.value
        ]

    @property
    def completion_percent(self) -> float:
        applicable = [
            i for i in self.items
            if i.status != CheckItemStatus.NOT_APPLICABLE
        ]
        if not applicable:
            return 0.0
        done = sum(
            1 for i in applicable
            if i.status in (
                CheckItemStatus.COMPLIANT,
                CheckItemStatus.NON_COMPLIANT,
                CheckItemStatus.WAIVED,
            )
        )
        return round(done / len(applicable) * 100, 2)

    def summary(self) -> dict:
        return {
            "checklist_id": self.checklist_id,
            "vehicle_model": self.vehicle_model,
            "manufacturer": self.manufacturer,
            "generated_at": self.generated_at,
            "total_items": self.total,
            "compliant": self.compliant_count,
            "non_compliant": self.non_compliant_count,
            "pending": self.pending_count,
            "not_applicable": sum(
                1 for i in self.items
                if i.status == CheckItemStatus.NOT_APPLICABLE
            ),
            "waived": sum(
                1 for i in self.items if i.status == CheckItemStatus.WAIVED
            ),
            "completion_percent": self.completion_percent,
            "critical_gaps": len(self.critical_non_compliant),
        }

    def to_dict(self) -> dict:
        return {
            "checklist_id": self.checklist_id,
            "vehicle_model": self.vehicle_model,
            "manufacturer": self.manufacturer,
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "items": [i.to_dict() for i in self.items],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def items_by_category(self, category: ClauseCategory) -> List[ChecklistItem]:
        return [i for i in self.items if i.category == category.value]

    def items_by_severity(self, severity: Severity) -> List[ChecklistItem]:
        return [i for i in self.items if i.severity == severity.value]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def generate_checklist(
    vehicle_model: str,
    manufacturer: str,
    checklist_id: Optional[str] = None,
    categories: Optional[List[ClauseCategory]] = None,
) -> Checklist:
    """
    Generate a fresh compliance checklist.

    Parameters
    ----------
    vehicle_model : str
        Identifier for the vehicle model / variant.
    manufacturer : str
        Name of the original equipment manufacturer.
    checklist_id : str, optional
        Custom checklist identifier; auto-generated if not provided.
    categories : list of ClauseCategory, optional
        Filter to specific clause categories. Defaults to all categories.

    Returns
    -------
    Checklist
        A pre-populated checklist with all applicable AIS-175 requirements.
    """
    if checklist_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        checklist_id = f"CL-AIS175-{ts}"

    if categories is not None:
        clauses = []
        for cat in categories:
            clauses.extend(get_clauses_by_category(cat))
    else:
        clauses = get_all_clauses()

    items: List[ChecklistItem] = []
    for clause in clauses:
        for idx, req in enumerate(clause.requirements, start=1):
            item_id = f"{clause.clause_id}-{idx:02d}"
            items.append(
                ChecklistItem(
                    item_id=item_id,
                    clause_id=clause.clause_id,
                    req_id=req.req_id,
                    title=f"[{clause.clause_id}] {clause.title}",
                    description=req.description,
                    mandatory=req.mandatory,
                    severity=clause.severity.value,
                    category=clause.category.value,
                )
            )

    return Checklist(
        checklist_id=checklist_id,
        vehicle_model=vehicle_model,
        manufacturer=manufacturer,
        items=items,
    )
