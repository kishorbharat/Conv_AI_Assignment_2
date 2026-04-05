"""
AIS-175 Compliance Engine – Demo Script
=========================================
Demonstrates the full end-to-end workflow:
  1. Create a homologation submission
  2. Validate an approval document (rule-based + ML risk scoring)
  3. Generate a compliance checklist
  4. Simulate manual review of checklist items
  5. Approve / reject the submission
  6. Export the audit trail
"""

import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from compliance_engine import (
    AuditTrail,
    CheckItemStatus,
    ClauseCategory,
    ComplianceEngine,
    Severity,
    get_all_clauses,
)

SEPARATOR = "=" * 70


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Sample approval document – realistic but partially incomplete (for demo)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT_COMPLETE = {
    # Homologation meta
    "vehicle_category": "M1",
    "ecu_list": ["ADAS-ECU", "Body-Control-Module", "Telematics-Unit"],
    "approval_authority": "Automotive Research Association of India (ARAI)",
    "approval_date": "2024-11-15",
    "homologation_date": "2024-11-15",
    # Software
    "software_version": "v3.2.1",
    "software_process": {
        "lifecycle_model": "V-Model",
        "coding_guidelines": "MISRA C:2012",
        "review_records": "SW-REV-2024-034",
    },
    "test_coverage_percent": 91.5,
    "test_execution_report": "TEST-EXEC-2024-088",
    # Cybersecurity
    "csms_policy": "CSMS-POLICY-v2.pdf",
    "cybersecurity_responsible_person": "Jane Doe",
    "tara": {
        "threat_list": ["Remote-code-execution", "CAN-bus-injection"],
        "risk_assessment": {"Remote-code-execution": "CAL-3", "CAN-bus-injection": "CAL-2"},
        "treatment_plan": "See TARA-2024-v3.pdf",
    },
    "cal_level": "CAL-3",
    "penetration_test_report": "PENTEST-2024-EXT-007.pdf",
    "ota_security_architecture": "OTA-ARCH-v1.docx",
    "code_signing_cert": "DigiCert-SHA256-RSA4096",
    # Functional Safety
    "hazop": {
        "hazard_list": ["Unintended-acceleration", "Loss-of-steering"],
        "risk_parameters": {"Unintended-acceleration": "ASIL-D"},
        "safety_goals": ["SG-01", "SG-02"],
    },
    "asil_level": "D",
    "safety_case_document": "SafetyCase-AIS175-2024.pdf",
    # Incident Response
    "incident_response_plan": "IRP-v2.pdf",
    "cvd_contact": "https://example-oem.in/security/disclosure",
    # Documentation
    "technical_doc_package": "TDP-XUV700-2024.zip",
    "doc_version": "3.0",
    "revision_history": ["v1.0 initial", "v2.0 TARA update", "v3.0 pentest added"],
    "audit_trail": "See compliance_engine audit log",
}

SAMPLE_DOCUMENT_INCOMPLETE = {
    # Missing many required fields to show failures
    "vehicle_category": "N1",
    "ecu_list": ["Engine-ECU"],
    "approval_authority": "ICAT",
    "approval_date": "2024-10-01",
    "homologation_date": "24/10/2024",  # Wrong format
    "software_version": "",              # Empty → will fail
    "test_coverage_percent": 72,         # Below 85% threshold
    # TARA incomplete
    "tara": {"threat_list": ["spoofing"]},
    "cal_level": "CAL-1",
    # HAZOP incomplete
    "hazop": {"hazard_list": ["rollover"]},
    "asil_level": "X",  # Invalid
}


def main() -> None:
    engine = ComplianceEngine(audit_trail=AuditTrail())

    # ==================================================================
    # SCENARIO 1 – Complete document → should PASS
    # ==================================================================
    print_section("SCENARIO 1: Complete / Compliant Submission")

    sub1 = engine.create_submission(
        vehicle_model="XUV-700",
        manufacturer="Mahindra & Mahindra",
        vehicle_category="M1",
        actor="admin@arai.gov.in",
    )
    print(f"  Submission ID : {sub1.submission_id}")

    report1 = engine.validate_document(
        submission_id=sub1.submission_id,
        document=SAMPLE_DOCUMENT_COMPLETE,
        document_id="DOC-2024-001",
        actor="validator@arai.gov.in",
    )
    s1 = report1.summary()
    print(f"  Validation ID : {s1['report_id']}")
    print(f"  Overall PASS  : {s1['overall_pass']}")
    print(f"  Requirements  : {s1['total_requirements']} total | "
          f"{s1['passed']} passed | {s1['failed']} failed")
    print(f"  Critical fails: {s1['critical_failures']}")
    print(f"  ML Risk Score : {s1['ml_risk_score']} ({s1['ml_risk_label']})")

    checklist1 = engine.generate_checklist(
        sub1.submission_id, actor="admin@arai.gov.in"
    )
    print(f"\n  Checklist ID  : {checklist1.checklist_id}")
    print(f"  Total items   : {checklist1.total}")

    # Simulate auto-populating checklist from validation results
    for item in checklist1.items:
        # Find the matching result
        match = next(
            (r for r in report1.results if r.req_id == item.req_id), None
        )
        if match:
            status = CheckItemStatus.COMPLIANT if match.passed else CheckItemStatus.NON_COMPLIANT
            engine.review_checklist_item(
                sub1.submission_id, item.item_id, status,
                reviewer="auto-validator", notes=match.failure_reason,
            )

    csum1 = checklist1.summary()
    print(f"  Completion    : {csum1['completion_percent']}%")
    print(f"  Compliant     : {csum1['compliant']}")
    print(f"  Non-compliant : {csum1['non_compliant']}")

    engine.approve_submission(
        sub1.submission_id,
        actor="approver@arai.gov.in",
        notes="All mandatory requirements satisfied.",
    )
    print(f"  Status        : {sub1.status.value}")

    # ==================================================================
    # SCENARIO 2 – Incomplete document → should FAIL
    # ==================================================================
    print_section("SCENARIO 2: Incomplete / Non-Compliant Submission")

    sub2 = engine.create_submission(
        vehicle_model="Cargo-Truck-3T",
        manufacturer="Tata Motors",
        vehicle_category="N1",
        actor="admin@icat.gov.in",
    )
    print(f"  Submission ID : {sub2.submission_id}")

    report2 = engine.validate_document(
        submission_id=sub2.submission_id,
        document=SAMPLE_DOCUMENT_INCOMPLETE,
        document_id="DOC-2024-002",
        actor="validator@icat.gov.in",
    )
    s2 = report2.summary()
    print(f"  Validation ID : {s2['report_id']}")
    print(f"  Overall PASS  : {s2['overall_pass']}")
    print(f"  Requirements  : {s2['total_requirements']} total | "
          f"{s2['passed']} passed | {s2['failed']} failed")
    print(f"  Critical fails: {s2['critical_failures']}")
    print(f"  ML Risk Score : {s2['ml_risk_score']} ({s2['ml_risk_label']})")

    print("\n  Critical / Major failures:")
    for r in report2.critical_failures + report2.major_failures:
        flag = "CRITICAL" if r.severity == "CRITICAL" else "MAJOR"
        print(f"    [{flag}] {r.req_id}: {r.failure_reason}")

    engine.reject_submission(
        sub2.submission_id,
        actor="approver@icat.gov.in",
        reason="Multiple critical non-conformances detected.",
    )
    print(f"\n  Status        : {sub2.status.value}")

    # ==================================================================
    # SCENARIO 3 – Clause-filtered checklist (Cybersecurity only)
    # ==================================================================
    print_section("SCENARIO 3: Category-Filtered Checklist (Cybersecurity only)")

    sub3 = engine.create_submission(
        vehicle_model="e-Versa",
        manufacturer="Honda Cars India",
        vehicle_category="M1",
        actor="admin@arai.gov.in",
    )
    cs_checklist = engine.generate_checklist(
        sub3.submission_id,
        actor="cs-reviewer@arai.gov.in",
        categories=[ClauseCategory.CYBERSECURITY],
    )
    print(f"  Checklist ID  : {cs_checklist.checklist_id}")
    print(f"  Filtered to   : Cybersecurity only")
    print(f"  Items         : {cs_checklist.total}")
    for item in cs_checklist.items:
        print(f"    [{item.severity}] {item.item_id}: {item.description[:60]}...")

    # ==================================================================
    # SCENARIO 4 – Dashboard & Audit Trail
    # ==================================================================
    print_section("SCENARIO 4: Compliance Dashboard & Audit Trail")

    dashboard = engine.get_compliance_dashboard()
    print(f"  Total submissions : {dashboard['total_submissions']}")
    print(f"  By status         : {dashboard['by_status']}")

    audit = engine.export_audit_trail()
    print(f"\n  Audit events logged: {len(audit['events'])}")
    print("  Last 5 events:")
    for ev in audit["events"][-5:]:
        print(f"    [{ev['timestamp'][:19]}] {ev['event_type']} "
              f"by {ev['actor']} → {ev['entity_id']}")

    # ==================================================================
    # Clause listing
    # ==================================================================
    print_section("AIS-175 Clause Registry (all clauses)")
    for clause in get_all_clauses():
        mand = sum(1 for r in clause.requirements if r.mandatory)
        opt = len(clause.requirements) - mand
        print(f"  [{clause.severity.value:14s}] Clause {clause.clause_id}: "
              f"{clause.title} "
              f"({mand} mandatory, {opt} optional)")

    print(f"\n{SEPARATOR}")
    print("  Demo complete. All scenarios executed successfully.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
