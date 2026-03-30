"""
Approval Document Validator
=============================
Validates homologation / type-approval submission documents against the
AIS-175 clause requirements encoded in ``clauses.py``.

The validator works in two passes:
1. **Schema pass** – checks that required evidence keys are present.
2. **Rule pass** – runs clause-specific validator callables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .clauses import (
    AIS175_CLAUSES,
    Clause,
    ClauseCategory,
    Requirement,
    Severity,
    get_all_clauses,
)


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------

@dataclass
class RequirementResult:
    """Outcome of validating one requirement."""

    req_id: str
    clause_id: str
    description: str
    passed: bool
    mandatory: bool
    severity: str
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Full validation report for an approval document."""

    report_id: str
    document_id: str
    vehicle_model: str
    manufacturer: str
    validated_at: str
    results: List[RequirementResult] = field(default_factory=list)
    ml_risk_score: Optional[float] = None
    ml_risk_label: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def passed(self) -> bool:
        """True only when all *mandatory* requirements pass."""
        return all(r.passed for r in self.results if r.mandatory)

    @property
    def critical_failures(self) -> List[RequirementResult]:
        return [
            r for r in self.results
            if not r.passed and r.severity == Severity.CRITICAL.value
        ]

    @property
    def major_failures(self) -> List[RequirementResult]:
        return [
            r for r in self.results
            if not r.passed and r.severity == Severity.MAJOR.value
        ]

    def summary(self) -> dict:
        total = len(self.results)
        passed_count = sum(1 for r in self.results if r.passed)
        return {
            "report_id": self.report_id,
            "document_id": self.document_id,
            "vehicle_model": self.vehicle_model,
            "manufacturer": self.manufacturer,
            "validated_at": self.validated_at,
            "overall_pass": self.passed,
            "total_requirements": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "critical_failures": len(self.critical_failures),
            "major_failures": len(self.major_failures),
            "ml_risk_score": self.ml_risk_score,
            "ml_risk_label": self.ml_risk_label,
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "requirement_results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# ML-assisted risk scorer
# ---------------------------------------------------------------------------

class MLRiskScorer:
    """
    Lightweight heuristic risk scorer that simulates ML-based risk estimation.

    In a production system this would be replaced by a trained classifier
    (e.g., scikit-learn / PyTorch model) that scores documents on their
    likelihood of post-approval non-conformances.

    Current heuristics
    ------------------
    - Missing critical evidence → high-risk contribution
    - Incomplete TARA / HAZOP → moderate-risk contribution
    - Low test coverage → moderate-risk contribution
    - Missing optional items → small-risk contribution
    """

    # Weight per evidence key absence
    WEIGHTS: Dict[str, float] = {
        "csms_policy": 0.15,
        "tara": 0.15,
        "hazop": 0.10,
        "asil_level": 0.10,
        "penetration_test_report": 0.08,
        "safety_case_document": 0.08,
        "test_coverage_percent": 0.07,
        "software_version": 0.05,
        "incident_response_plan": 0.06,
        "technical_doc_package": 0.06,
        "approval_authority": 0.05,
        "audit_trail": 0.05,
    }

    def score(self, doc: dict) -> Tuple[float, str]:
        """
        Return (risk_score, risk_label) where risk_score ∈ [0.0, 1.0].

        Labels: LOW (<0.25), MEDIUM (0.25–0.60), HIGH (>0.60).
        """
        risk = 0.0
        for key, weight in self.WEIGHTS.items():
            if not doc.get(key):
                risk += weight

        # Penalise low test coverage
        cov = float(doc.get("test_coverage_percent", 0) or 0)
        if cov < 85:
            shortfall = (85 - cov) / 85
            risk += shortfall * 0.07

        risk = min(risk, 1.0)

        if risk < 0.25:
            label = "LOW"
        elif risk < 0.60:
            label = "MEDIUM"
        else:
            label = "HIGH"

        return round(risk, 4), label


# ---------------------------------------------------------------------------
# Document Validator
# ---------------------------------------------------------------------------

class DocumentValidator:
    """
    Validates an approval document dict against AIS-175 requirements.

    Parameters
    ----------
    use_ml : bool
        When True, also compute an ML-based risk score.
    """

    def __init__(self, use_ml: bool = True) -> None:
        self._ml_scorer = MLRiskScorer() if use_ml else None

    def validate(
        self,
        document: dict,
        document_id: str = "DOC-001",
        vehicle_model: str = "Unknown",
        manufacturer: str = "Unknown",
        report_id: Optional[str] = None,
    ) -> ValidationReport:
        """
        Validate *document* against all AIS-175 clauses.

        Parameters
        ----------
        document : dict
            Approval submission data.  Keys correspond to ``evidence_keys``
            declared on each :class:`~compliance_engine.clauses.Requirement`.
        document_id : str
            Reference identifier for the document being validated.
        vehicle_model : str
            Vehicle model name / variant.
        manufacturer : str
            OEM name.
        report_id : str, optional
            Custom report ID; auto-generated if not supplied.

        Returns
        -------
        ValidationReport
        """
        if report_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            report_id = f"VR-AIS175-{ts}"

        results: List[RequirementResult] = []

        for clause in get_all_clauses():
            for req in clause.requirements:
                result = self._check_requirement(clause, req, document)
                results.append(result)

        ml_score: Optional[float] = None
        ml_label: Optional[str] = None
        if self._ml_scorer is not None:
            ml_score, ml_label = self._ml_scorer.score(document)

        return ValidationReport(
            report_id=report_id,
            document_id=document_id,
            vehicle_model=vehicle_model,
            manufacturer=manufacturer,
            validated_at=datetime.now(timezone.utc).isoformat(),
            results=results,
            ml_risk_score=ml_score,
            ml_risk_label=ml_label,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_requirement(
        clause: Clause,
        req: Requirement,
        doc: dict,
    ) -> RequirementResult:
        """Evaluate a single requirement against *doc*."""
        # 1. Schema pass – check evidence keys are present
        missing_keys = [k for k in req.evidence_keys if not doc.get(k)]
        if missing_keys:
            return RequirementResult(
                req_id=req.req_id,
                clause_id=clause.clause_id,
                description=req.description,
                passed=False,
                mandatory=req.mandatory,
                severity=clause.severity.value,
                failure_reason=f"Missing evidence keys: {missing_keys}",
            )

        # 2. Rule pass – run custom validator if defined
        if req.validator is not None:
            try:
                ok = req.validator(doc)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                ok = False
                return RequirementResult(
                    req_id=req.req_id,
                    clause_id=clause.clause_id,
                    description=req.description,
                    passed=False,
                    mandatory=req.mandatory,
                    severity=clause.severity.value,
                    failure_reason=f"Validator raised exception: {exc}",
                )
            if not ok:
                return RequirementResult(
                    req_id=req.req_id,
                    clause_id=clause.clause_id,
                    description=req.description,
                    passed=False,
                    mandatory=req.mandatory,
                    severity=clause.severity.value,
                    failure_reason="Custom rule validation failed.",
                )

        return RequirementResult(
            req_id=req.req_id,
            clause_id=clause.clause_id,
            description=req.description,
            passed=True,
            mandatory=req.mandatory,
            severity=clause.severity.value,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_document(
    document: dict,
    document_id: str = "DOC-001",
    vehicle_model: str = "Unknown",
    manufacturer: str = "Unknown",
    use_ml: bool = True,
) -> ValidationReport:
    """Module-level shortcut for one-shot document validation."""
    validator = DocumentValidator(use_ml=use_ml)
    return validator.validate(
        document=document,
        document_id=document_id,
        vehicle_model=vehicle_model,
        manufacturer=manufacturer,
    )
