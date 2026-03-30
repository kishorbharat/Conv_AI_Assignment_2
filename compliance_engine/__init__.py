"""
AIS-175 Compliance Engine
==========================
Regulatory Knowledge Engineering & Compliance Automation for AIS-175.

Quick-start
-----------
>>> from compliance_engine import ComplianceEngine
>>> engine = ComplianceEngine()
>>> sub = engine.create_submission("XUV-700", "Mahindra", "M1")
>>> report = engine.validate_document(sub.submission_id, document={...})
>>> checklist = engine.generate_checklist(sub.submission_id)
"""

from .engine import ComplianceEngine, Submission, SubmissionStatus
from .checklist import Checklist, ChecklistItem, CheckItemStatus, generate_checklist
from .validator import DocumentValidator, ValidationReport, validate_document
from .audit import AuditTrail, AuditEvent, AuditEventType, default_trail
from .clauses import (
    AIS175_CLAUSES,
    Clause,
    ClauseCategory,
    Requirement,
    Severity,
    get_all_clauses,
    get_clauses_by_category,
    get_mandatory_requirements,
)

__all__ = [
    # Engine
    "ComplianceEngine",
    "Submission",
    "SubmissionStatus",
    # Checklist
    "Checklist",
    "ChecklistItem",
    "CheckItemStatus",
    "generate_checklist",
    # Validator
    "DocumentValidator",
    "ValidationReport",
    "validate_document",
    # Audit
    "AuditTrail",
    "AuditEvent",
    "AuditEventType",
    "default_trail",
    # Clauses
    "AIS175_CLAUSES",
    "Clause",
    "ClauseCategory",
    "Requirement",
    "Severity",
    "get_all_clauses",
    "get_clauses_by_category",
    "get_mandatory_requirements",
]
