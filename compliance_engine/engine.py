"""
AIS-175 Compliance Engine – Core Orchestrator
===============================================
The ``ComplianceEngine`` ties together clause definitions, checklist
generation, document validation, and audit-trail logging into a single
high-level API.

Typical workflow
----------------
1. Create a ``ComplianceEngine`` instance.
2. Call :meth:`create_submission` with vehicle meta-data.
3. Call :meth:`validate_document` to run automated checks.
4. Call :meth:`generate_checklist` to get a reviewable checklist.
5. Optionally update checklist items via :meth:`review_checklist_item`.
6. Call :meth:`approve_submission` or :meth:`reject_submission`.
7. Export audit trail via :meth:`export_audit_trail`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from .audit import AuditEventType, AuditTrail, default_trail
from .checklist import (
    Checklist,
    ChecklistItem,
    CheckItemStatus,
    ClauseCategory,
    generate_checklist,
)
from .validator import DocumentValidator, ValidationReport, validate_document


class SubmissionStatus(str, Enum):
    DRAFT = "DRAFT"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"


@dataclass
class Submission:
    """Represents a single homologation submission for a vehicle variant."""

    submission_id: str
    vehicle_model: str
    manufacturer: str
    vehicle_category: str
    created_at: str
    status: SubmissionStatus = SubmissionStatus.DRAFT
    checklist: Optional[Checklist] = field(default=None, repr=False)
    validation_report: Optional[ValidationReport] = field(default=None, repr=False)
    documents: Dict[str, dict] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        d = {
            "submission_id": self.submission_id,
            "vehicle_model": self.vehicle_model,
            "manufacturer": self.manufacturer,
            "vehicle_category": self.vehicle_category,
            "created_at": self.created_at,
            "status": self.status.value,
            "notes": self.notes,
        }
        if self.checklist:
            d["checklist_summary"] = self.checklist.summary()
        if self.validation_report:
            d["validation_summary"] = self.validation_report.summary()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ComplianceEngine:
    """
    High-level AIS-175 compliance orchestrator.

    Parameters
    ----------
    audit_trail : AuditTrail, optional
        Shared audit trail instance. Uses the module-level ``default_trail``
        if not provided.
    """

    def __init__(self, audit_trail: Optional[AuditTrail] = None) -> None:
        self._audit = audit_trail or default_trail
        self._submissions: Dict[str, Submission] = {}

    # ------------------------------------------------------------------
    # Submission management
    # ------------------------------------------------------------------

    def create_submission(
        self,
        vehicle_model: str,
        manufacturer: str,
        vehicle_category: str,
        actor: str = "system",
    ) -> Submission:
        """Create a new homologation submission."""
        submission_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"
        sub = Submission(
            submission_id=submission_id,
            vehicle_model=vehicle_model,
            manufacturer=manufacturer,
            vehicle_category=vehicle_category,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._submissions[submission_id] = sub
        self._audit.log(
            AuditEventType.SUBMISSION_CREATED,
            actor=actor,
            entity_id=submission_id,
            entity_type="Submission",
            details={
                "vehicle_model": vehicle_model,
                "manufacturer": manufacturer,
                "vehicle_category": vehicle_category,
            },
        )
        return sub

    def get_submission(self, submission_id: str) -> Optional[Submission]:
        return self._submissions.get(submission_id)

    # ------------------------------------------------------------------
    # Checklist
    # ------------------------------------------------------------------

    def generate_checklist(
        self,
        submission_id: str,
        actor: str = "system",
        categories: Optional[List[ClauseCategory]] = None,
    ) -> Checklist:
        """Generate and attach a compliance checklist to the submission."""
        sub = self._require_submission(submission_id)
        checklist = generate_checklist(
            vehicle_model=sub.vehicle_model,
            manufacturer=sub.manufacturer,
            categories=categories,
        )
        sub.checklist = checklist
        self._audit.log_checklist_generated(
            actor=actor,
            checklist_id=checklist.checklist_id,
            vehicle_model=sub.vehicle_model,
            manufacturer=sub.manufacturer,
        )
        return checklist

    def review_checklist_item(
        self,
        submission_id: str,
        item_id: str,
        status: CheckItemStatus,
        reviewer: str,
        notes: str = "",
    ) -> ChecklistItem:
        """Update the status of a single checklist item."""
        sub = self._require_submission(submission_id)
        if sub.checklist is None:
            raise ValueError(
                f"Submission {submission_id} has no checklist. "
                "Call generate_checklist() first."
            )
        item = next(
            (i for i in sub.checklist.items if i.item_id == item_id), None
        )
        if item is None:
            raise ValueError(f"Checklist item '{item_id}' not found.")

        if status == CheckItemStatus.COMPLIANT:
            item.mark_compliant(reviewer=reviewer, notes=notes)
        elif status == CheckItemStatus.NON_COMPLIANT:
            item.mark_non_compliant(reviewer=reviewer, notes=notes)
        elif status == CheckItemStatus.NOT_APPLICABLE:
            item.mark_not_applicable(reason=notes)
        elif status == CheckItemStatus.WAIVED:
            item.waive(reason=notes, reviewer=reviewer)

        self._audit.log_item_reviewed(
            actor=reviewer,
            checklist_id=sub.checklist.checklist_id,
            item_id=item_id,
            status=status.value,
            notes=notes,
        )
        return item

    # ------------------------------------------------------------------
    # Document validation
    # ------------------------------------------------------------------

    def validate_document(
        self,
        submission_id: str,
        document: dict,
        document_id: str = "DOC-001",
        actor: str = "system",
        use_ml: bool = True,
    ) -> ValidationReport:
        """Validate an approval document and attach the report to the submission."""
        sub = self._require_submission(submission_id)
        sub.documents[document_id] = document
        sub.status = SubmissionStatus.UNDER_REVIEW

        report = validate_document(
            document=document,
            document_id=document_id,
            vehicle_model=sub.vehicle_model,
            manufacturer=sub.manufacturer,
            use_ml=use_ml,
        )
        sub.validation_report = report

        self._audit.log_document_validated(
            actor=actor,
            report_id=report.report_id,
            document_id=document_id,
            overall_pass=report.passed,
            ml_risk_label=report.ml_risk_label,
        )
        return report

    # ------------------------------------------------------------------
    # Approval / rejection
    # ------------------------------------------------------------------

    def approve_submission(
        self, submission_id: str, actor: str, notes: str = ""
    ) -> Submission:
        """Mark a submission as approved."""
        sub = self._require_submission(submission_id)
        sub.status = SubmissionStatus.APPROVED
        if notes:
            sub.notes = notes
        self._audit.log_submission_approved(
            actor=actor,
            submission_id=submission_id,
            vehicle_model=sub.vehicle_model,
        )
        return sub

    def reject_submission(
        self, submission_id: str, actor: str, reason: str
    ) -> Submission:
        """Mark a submission as rejected."""
        sub = self._require_submission(submission_id)
        sub.status = SubmissionStatus.REJECTED
        sub.notes = reason
        self._audit.log_submission_rejected(
            actor=actor,
            submission_id=submission_id,
            reason=reason,
        )
        return sub

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def export_audit_trail(self) -> dict:
        return self._audit.to_dict()

    def export_audit_trail_json(self, indent: int = 2) -> str:
        return self._audit.to_json(indent=indent)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_compliance_dashboard(self) -> dict:
        """Return a summary dashboard across all submissions."""
        total = len(self._submissions)
        by_status: Dict[str, int] = {}
        for sub in self._submissions.values():
            key = sub.status.value
            by_status[key] = by_status.get(key, 0) + 1

        return {
            "total_submissions": total,
            "by_status": by_status,
            "submissions": [s.to_dict() for s in self._submissions.values()],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_submission(self, submission_id: str) -> Submission:
        sub = self._submissions.get(submission_id)
        if sub is None:
            raise ValueError(f"Submission '{submission_id}' not found.")
        return sub
