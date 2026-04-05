"""
AIS-175 Clause Definitions
===========================
AIS-175 (Automotive Industry Standard 175) governs requirements for
Automotive Electronic Control Units (ECUs) and software, including
cybersecurity, functional safety, and homologation requirements for
vehicles sold in India.

This module encodes the clause-wise requirements so they can be
programmatically evaluated by the compliance engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class Severity(str, Enum):
    """Risk / non-compliance severity levels."""
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFORMATIONAL = "INFORMATIONAL"


class ClauseCategory(str, Enum):
    """Top-level categories that group AIS-175 clauses."""
    CYBERSECURITY = "Cybersecurity"
    FUNCTIONAL_SAFETY = "Functional Safety"
    SOFTWARE_PROCESS = "Software Process"
    HOMOLOGATION = "Homologation"
    DOCUMENTATION = "Documentation"
    TESTING = "Testing"
    INCIDENT_RESPONSE = "Incident Response"


@dataclass
class Requirement:
    """A single verifiable requirement extracted from an AIS-175 clause."""

    req_id: str
    description: str
    mandatory: bool = True
    evidence_keys: List[str] = field(default_factory=list)
    """Keys that must be present in the document under review."""

    validator: Optional[Callable[[dict], bool]] = field(default=None, repr=False)
    """Optional callable(doc_data) -> bool for custom validation logic."""


@dataclass
class Clause:
    """Represents a single numbered clause within AIS-175."""

    clause_id: str
    title: str
    category: ClauseCategory
    severity: Severity
    description: str
    requirements: List[Requirement] = field(default_factory=list)
    sub_clauses: List["Clause"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper validators (reusable lambdas / functions)
# ---------------------------------------------------------------------------

def _has_version(doc: dict) -> bool:
    v = doc.get("software_version", "")
    return bool(v) and v != "N/A"


def _has_valid_date(doc: dict) -> bool:
    import re
    date = doc.get("homologation_date", "")
    return bool(re.match(r"\d{4}-\d{2}-\d{2}", str(date)))


def _tara_complete(doc: dict) -> bool:
    tara = doc.get("tara", {})
    return all(k in tara for k in ("threat_list", "risk_assessment", "treatment_plan"))


def _has_pen_test(doc: dict) -> bool:
    return bool(doc.get("penetration_test_report"))


def _hazop_complete(doc: dict) -> bool:
    hazop = doc.get("hazop", {})
    return all(k in hazop for k in ("hazard_list", "risk_parameters", "safety_goals"))


def _asil_defined(doc: dict) -> bool:
    return doc.get("asil_level") in ("QM", "A", "B", "C", "D")


def _sw_process_documented(doc: dict) -> bool:
    sp = doc.get("software_process", {})
    return all(k in sp for k in ("lifecycle_model", "coding_guidelines", "review_records"))


def _test_coverage(doc: dict) -> bool:
    cov = doc.get("test_coverage_percent", 0)
    return float(cov) >= 85.0


def _incident_plan_exists(doc: dict) -> bool:
    return bool(doc.get("incident_response_plan"))


def _approval_authority_present(doc: dict) -> bool:
    return bool(doc.get("approval_authority")) and bool(doc.get("approval_date"))


# ---------------------------------------------------------------------------
# AIS-175 Clause Registry
# ---------------------------------------------------------------------------

AIS175_CLAUSES: Dict[str, Clause] = {
    # ------------------------------------------------------------------ #
    # Clause 4 – Scope & General Requirements
    # ------------------------------------------------------------------ #
    "4.1": Clause(
        clause_id="4.1",
        title="Scope of Application",
        category=ClauseCategory.HOMOLOGATION,
        severity=Severity.CRITICAL,
        description=(
            "Defines the vehicles and ECU categories to which AIS-175 applies, "
            "including M-category passenger vehicles, N-category goods vehicles, "
            "and L-category two/three wheelers equipped with connected ECUs."
        ),
        requirements=[
            Requirement(
                req_id="4.1-R1",
                description="Vehicle category must be declared (M1/M2/N1/N2/L3/L5 etc.).",
                evidence_keys=["vehicle_category"],
            ),
            Requirement(
                req_id="4.1-R2",
                description="Applicable ECU list must be provided.",
                evidence_keys=["ecu_list"],
            ),
        ],
    ),
    "4.2": Clause(
        clause_id="4.2",
        title="Type Approval Prerequisites",
        category=ClauseCategory.HOMOLOGATION,
        severity=Severity.CRITICAL,
        description=(
            "Manufacturer must demonstrate compliance before type-approval "
            "certificate is issued by the testing agency."
        ),
        requirements=[
            Requirement(
                req_id="4.2-R1",
                description="Approval authority name and date must be present.",
                evidence_keys=["approval_authority", "approval_date"],
                validator=_approval_authority_present,
            ),
            Requirement(
                req_id="4.2-R2",
                description="Homologation date must follow ISO 8601 (YYYY-MM-DD).",
                evidence_keys=["homologation_date"],
                validator=_has_valid_date,
            ),
        ],
    ),
    # ------------------------------------------------------------------ #
    # Clause 5 – Cybersecurity Management
    # ------------------------------------------------------------------ #
    "5.1": Clause(
        clause_id="5.1",
        title="Cybersecurity Management System (CSMS)",
        category=ClauseCategory.CYBERSECURITY,
        severity=Severity.CRITICAL,
        description=(
            "Manufacturer must implement a CSMS aligned with ISO/SAE 21434 "
            "covering the full vehicle lifecycle from concept to decommissioning."
        ),
        requirements=[
            Requirement(
                req_id="5.1-R1",
                description="CSMS policy document must be present.",
                evidence_keys=["csms_policy"],
            ),
            Requirement(
                req_id="5.1-R2",
                description="Cybersecurity responsibilities must be defined.",
                evidence_keys=["cybersecurity_responsible_person"],
            ),
        ],
    ),
    "5.2": Clause(
        clause_id="5.2",
        title="Threat Analysis and Risk Assessment (TARA)",
        category=ClauseCategory.CYBERSECURITY,
        severity=Severity.CRITICAL,
        description=(
            "A TARA must be performed for each identified asset. "
            "Outputs include threat list, risk ratings (CAL 1-4), and treatment plans."
        ),
        requirements=[
            Requirement(
                req_id="5.2-R1",
                description="TARA must include threat_list, risk_assessment, and treatment_plan.",
                evidence_keys=["tara"],
                validator=_tara_complete,
            ),
            Requirement(
                req_id="5.2-R2",
                description="Cybersecurity Assurance Level (CAL) must be assigned.",
                evidence_keys=["cal_level"],
            ),
        ],
    ),
    "5.3": Clause(
        clause_id="5.3",
        title="Penetration Testing",
        category=ClauseCategory.CYBERSECURITY,
        severity=Severity.MAJOR,
        description=(
            "Penetration testing must be conducted by a qualified third party "
            "before type approval. Report must be submitted to the testing agency."
        ),
        requirements=[
            Requirement(
                req_id="5.3-R1",
                description="Penetration test report reference must be provided.",
                evidence_keys=["penetration_test_report"],
                validator=_has_pen_test,
            ),
        ],
    ),
    "5.4": Clause(
        clause_id="5.4",
        title="Over-the-Air (OTA) Update Security",
        category=ClauseCategory.CYBERSECURITY,
        severity=Severity.MAJOR,
        description=(
            "Where OTA updates are supported, a secure update mechanism with "
            "cryptographic signing and rollback protection must be implemented."
        ),
        requirements=[
            Requirement(
                req_id="5.4-R1",
                description="OTA security architecture document must be provided if OTA is supported.",
                mandatory=False,
                evidence_keys=["ota_security_architecture"],
            ),
            Requirement(
                req_id="5.4-R2",
                description="Code signing certificate details must be listed.",
                mandatory=False,
                evidence_keys=["code_signing_cert"],
            ),
        ],
    ),
    # ------------------------------------------------------------------ #
    # Clause 6 – Functional Safety
    # ------------------------------------------------------------------ #
    "6.1": Clause(
        clause_id="6.1",
        title="HAZOP / HARA",
        category=ClauseCategory.FUNCTIONAL_SAFETY,
        severity=Severity.CRITICAL,
        description=(
            "Hazard Analysis and Risk Assessment (HARA/HAZOP) must be performed "
            "for all safety-related items. ASIL decomposition must be documented."
        ),
        requirements=[
            Requirement(
                req_id="6.1-R1",
                description="HAZOP/HARA must include hazard_list, risk_parameters, and safety_goals.",
                evidence_keys=["hazop"],
                validator=_hazop_complete,
            ),
        ],
    ),
    "6.2": Clause(
        clause_id="6.2",
        title="ASIL Classification",
        category=ClauseCategory.FUNCTIONAL_SAFETY,
        severity=Severity.CRITICAL,
        description=(
            "Every safety-related function must be assigned an ASIL level "
            "(QM / A / B / C / D) according to ISO 26262."
        ),
        requirements=[
            Requirement(
                req_id="6.2-R1",
                description="ASIL level must be one of QM, A, B, C, or D.",
                evidence_keys=["asil_level"],
                validator=_asil_defined,
            ),
        ],
    ),
    "6.3": Clause(
        clause_id="6.3",
        title="Safety Case",
        category=ClauseCategory.FUNCTIONAL_SAFETY,
        severity=Severity.CRITICAL,
        description=(
            "A safety case document linking each safety goal to its validation "
            "evidence must be maintained throughout the product lifecycle."
        ),
        requirements=[
            Requirement(
                req_id="6.3-R1",
                description="Safety case document must be present.",
                evidence_keys=["safety_case_document"],
            ),
        ],
    ),
    # ------------------------------------------------------------------ #
    # Clause 7 – Software Development Process
    # ------------------------------------------------------------------ #
    "7.1": Clause(
        clause_id="7.1",
        title="Software Lifecycle & Coding Guidelines",
        category=ClauseCategory.SOFTWARE_PROCESS,
        severity=Severity.MAJOR,
        description=(
            "Software must be developed using a documented lifecycle model "
            "(e.g., V-model). Coding guidelines (e.g., MISRA C) must be followed."
        ),
        requirements=[
            Requirement(
                req_id="7.1-R1",
                description="Software process must include lifecycle_model, coding_guidelines, review_records.",
                evidence_keys=["software_process"],
                validator=_sw_process_documented,
            ),
            Requirement(
                req_id="7.1-R2",
                description="Software version identifier must be present.",
                evidence_keys=["software_version"],
                validator=_has_version,
            ),
        ],
    ),
    "7.2": Clause(
        clause_id="7.2",
        title="Software Testing & Coverage",
        category=ClauseCategory.TESTING,
        severity=Severity.MAJOR,
        description=(
            "Unit, integration, and system-level tests must be executed. "
            "MC/DC coverage ≥ 85% required for ASIL B/C/D items."
        ),
        requirements=[
            Requirement(
                req_id="7.2-R1",
                description="Test coverage must be ≥ 85%.",
                evidence_keys=["test_coverage_percent"],
                validator=_test_coverage,
            ),
            Requirement(
                req_id="7.2-R2",
                description="Test execution report must be present.",
                evidence_keys=["test_execution_report"],
            ),
        ],
    ),
    # ------------------------------------------------------------------ #
    # Clause 8 – Incident Response & Vulnerability Management
    # ------------------------------------------------------------------ #
    "8.1": Clause(
        clause_id="8.1",
        title="Incident Response Plan",
        category=ClauseCategory.INCIDENT_RESPONSE,
        severity=Severity.MAJOR,
        description=(
            "Manufacturer must have a documented incident response plan covering "
            "detection, triage, notification, and remediation."
        ),
        requirements=[
            Requirement(
                req_id="8.1-R1",
                description="Incident response plan must be referenced.",
                evidence_keys=["incident_response_plan"],
                validator=_incident_plan_exists,
            ),
        ],
    ),
    "8.2": Clause(
        clause_id="8.2",
        title="Vulnerability Disclosure",
        category=ClauseCategory.INCIDENT_RESPONSE,
        severity=Severity.MINOR,
        description=(
            "A publicly accessible coordinated vulnerability disclosure (CVD) "
            "policy / contact must be provided."
        ),
        requirements=[
            Requirement(
                req_id="8.2-R1",
                description="CVD policy or contact URL must be provided.",
                mandatory=False,
                evidence_keys=["cvd_contact"],
            ),
        ],
    ),
    # ------------------------------------------------------------------ #
    # Clause 9 – Documentation & Record Keeping
    # ------------------------------------------------------------------ #
    "9.1": Clause(
        clause_id="9.1",
        title="Technical Documentation Package",
        category=ClauseCategory.DOCUMENTATION,
        severity=Severity.CRITICAL,
        description=(
            "A complete technical documentation package must be submitted to the "
            "testing agency. It must be kept up to date for the vehicle's lifetime."
        ),
        requirements=[
            Requirement(
                req_id="9.1-R1",
                description="Technical documentation package reference must be provided.",
                evidence_keys=["technical_doc_package"],
            ),
            Requirement(
                req_id="9.1-R2",
                description="Document version and revision history must be present.",
                evidence_keys=["doc_version", "revision_history"],
            ),
        ],
    ),
    "9.2": Clause(
        clause_id="9.2",
        title="Audit Trail",
        category=ClauseCategory.DOCUMENTATION,
        severity=Severity.MAJOR,
        description=(
            "All compliance activities, reviews, test results, and approvals must "
            "be traceable through an audit trail with timestamps and responsible persons."
        ),
        requirements=[
            Requirement(
                req_id="9.2-R1",
                description="Audit trail log must reference responsible persons and timestamps.",
                evidence_keys=["audit_trail"],
            ),
        ],
    ),
}


def get_all_clauses() -> List[Clause]:
    """Return all registered AIS-175 clauses in clause-ID order."""
    return [AIS175_CLAUSES[k] for k in sorted(AIS175_CLAUSES)]


def get_clauses_by_category(category: ClauseCategory) -> List[Clause]:
    """Return all clauses belonging to a specific category."""
    return [c for c in AIS175_CLAUSES.values() if c.category == category]


def get_mandatory_requirements() -> List[Requirement]:
    """Return every mandatory requirement across all clauses."""
    reqs: List[Requirement] = []
    for clause in AIS175_CLAUSES.values():
        reqs.extend(r for r in clause.requirements if r.mandatory)
    return reqs
