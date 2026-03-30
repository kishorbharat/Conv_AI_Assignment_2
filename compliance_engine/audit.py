"""
Audit Trail & Traceability
===========================
Provides an append-only audit log that records every compliance action,
review decision, validation run, and approval event for high audit traceability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class AuditEventType(str, Enum):
    CHECKLIST_GENERATED = "CHECKLIST_GENERATED"
    DOCUMENT_VALIDATED = "DOCUMENT_VALIDATED"
    ITEM_REVIEWED = "ITEM_REVIEWED"
    ITEM_WAIVED = "ITEM_WAIVED"
    REPORT_APPROVED = "REPORT_APPROVED"
    REPORT_REJECTED = "REPORT_REJECTED"
    CLAUSE_UPDATED = "CLAUSE_UPDATED"
    SUBMISSION_CREATED = "SUBMISSION_CREATED"
    SUBMISSION_APPROVED = "SUBMISSION_APPROVED"
    SUBMISSION_REJECTED = "SUBMISSION_REJECTED"


@dataclass
class AuditEvent:
    """Single immutable entry in the audit trail."""

    event_id: str
    event_type: AuditEventType
    timestamp: str
    actor: str
    entity_id: str
    entity_type: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class AuditTrail:
    """
    Thread-safe (append-only) in-memory audit trail.

    In production, ``_events`` would be persisted to a database or
    immutable log store (e.g., AWS CloudTrail, Azure Monitor, or a
    blockchain-anchored ledger).
    """

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        entity_id: str,
        entity_type: str,
        details: Optional[dict] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            entity_id=entity_id,
            entity_type=entity_type,
            details=details or {},
        )
        self._events.append(event)
        return event

    def log_checklist_generated(
        self, actor: str, checklist_id: str, vehicle_model: str, manufacturer: str
    ) -> AuditEvent:
        return self.log(
            AuditEventType.CHECKLIST_GENERATED,
            actor=actor,
            entity_id=checklist_id,
            entity_type="Checklist",
            details={"vehicle_model": vehicle_model, "manufacturer": manufacturer},
        )

    def log_document_validated(
        self,
        actor: str,
        report_id: str,
        document_id: str,
        overall_pass: bool,
        ml_risk_label: Optional[str] = None,
    ) -> AuditEvent:
        return self.log(
            AuditEventType.DOCUMENT_VALIDATED,
            actor=actor,
            entity_id=report_id,
            entity_type="ValidationReport",
            details={
                "document_id": document_id,
                "overall_pass": overall_pass,
                "ml_risk_label": ml_risk_label,
            },
        )

    def log_item_reviewed(
        self,
        actor: str,
        checklist_id: str,
        item_id: str,
        status: str,
        notes: str = "",
    ) -> AuditEvent:
        return self.log(
            AuditEventType.ITEM_REVIEWED,
            actor=actor,
            entity_id=checklist_id,
            entity_type="ChecklistItem",
            details={"item_id": item_id, "status": status, "notes": notes},
        )

    def log_submission_approved(
        self, actor: str, submission_id: str, vehicle_model: str
    ) -> AuditEvent:
        return self.log(
            AuditEventType.SUBMISSION_APPROVED,
            actor=actor,
            entity_id=submission_id,
            entity_type="Submission",
            details={"vehicle_model": vehicle_model},
        )

    def log_submission_rejected(
        self, actor: str, submission_id: str, reason: str
    ) -> AuditEvent:
        return self.log(
            AuditEventType.SUBMISSION_REJECTED,
            actor=actor,
            entity_id=submission_id,
            entity_type="Submission",
            details={"reason": reason},
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def events(self) -> List[AuditEvent]:
        return list(self._events)

    def events_for_entity(self, entity_id: str) -> List[AuditEvent]:
        return [e for e in self._events if e.entity_id == entity_id]

    def events_by_actor(self, actor: str) -> List[AuditEvent]:
        return [e for e in self._events if e.actor == actor]

    def events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def to_dict(self) -> dict:
        return {"events": [e.to_dict() for e in self._events]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# Module-level default trail (can be replaced or injected)
default_trail = AuditTrail()
