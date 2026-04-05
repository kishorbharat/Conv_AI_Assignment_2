"""
Tests for the AIS-175 Compliance Engine
"""

from __future__ import annotations

import pytest

from compliance_engine import (
    AIS175_CLAUSES,
    AuditEventType,
    AuditTrail,
    CheckItemStatus,
    Checklist,
    ClauseCategory,
    ComplianceEngine,
    Severity,
    SubmissionStatus,
    ValidationReport,
    generate_checklist,
    get_all_clauses,
    get_clauses_by_category,
    get_mandatory_requirements,
    validate_document,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPLETE_DOCUMENT = {
    "vehicle_category": "M1",
    "ecu_list": ["ADAS-ECU"],
    "approval_authority": "ARAI",
    "approval_date": "2024-11-15",
    "homologation_date": "2024-11-15",
    "software_version": "v1.0.0",
    "software_process": {
        "lifecycle_model": "V-Model",
        "coding_guidelines": "MISRA C:2012",
        "review_records": "REV-001",
    },
    "test_coverage_percent": 90,
    "test_execution_report": "TEST-001",
    "csms_policy": "CSMS-v1.pdf",
    "cybersecurity_responsible_person": "Alice",
    "tara": {
        "threat_list": ["injection"],
        "risk_assessment": {"injection": "CAL-2"},
        "treatment_plan": "treat.pdf",
    },
    "cal_level": "CAL-2",
    "penetration_test_report": "PENTEST-001.pdf",
    "hazop": {
        "hazard_list": ["loss-of-control"],
        "risk_parameters": {"loss-of-control": "ASIL-C"},
        "safety_goals": ["SG-01"],
    },
    "asil_level": "C",
    "safety_case_document": "SafetyCase.pdf",
    "incident_response_plan": "IRP.pdf",
    "technical_doc_package": "TDP.zip",
    "doc_version": "1.0",
    "revision_history": ["v1.0 initial"],
    "audit_trail": "trail.log",
}

INCOMPLETE_DOCUMENT = {
    "vehicle_category": "N1",
    "ecu_list": ["Engine-ECU"],
}


@pytest.fixture()
def engine() -> ComplianceEngine:
    return ComplianceEngine(audit_trail=AuditTrail())


# ---------------------------------------------------------------------------
# 1. Clause registry tests
# ---------------------------------------------------------------------------

class TestClauseRegistry:
    def test_clauses_loaded(self):
        assert len(AIS175_CLAUSES) > 0

    def test_all_clauses_have_requirements(self):
        for clause in AIS175_CLAUSES.values():
            assert len(clause.requirements) > 0, (
                f"Clause {clause.clause_id} has no requirements"
            )

    def test_clause_ids_sorted(self):
        ids = list(AIS175_CLAUSES.keys())
        assert ids == sorted(ids)

    def test_get_all_clauses_sorted(self):
        clauses = get_all_clauses()
        ids = [c.clause_id for c in clauses]
        assert ids == sorted(ids)

    def test_get_clauses_by_category(self):
        cyber = get_clauses_by_category(ClauseCategory.CYBERSECURITY)
        assert all(c.category == ClauseCategory.CYBERSECURITY for c in cyber)
        assert len(cyber) > 0

    def test_get_mandatory_requirements(self):
        mandatory = get_mandatory_requirements()
        assert all(r.mandatory for r in mandatory)
        assert len(mandatory) > 0

    def test_severity_values(self):
        for clause in AIS175_CLAUSES.values():
            assert clause.severity in list(Severity)

    def test_category_values(self):
        for clause in AIS175_CLAUSES.values():
            assert clause.category in list(ClauseCategory)


# ---------------------------------------------------------------------------
# 2. Checklist generation tests
# ---------------------------------------------------------------------------

class TestChecklistGeneration:
    def test_generates_checklist(self):
        cl = generate_checklist("Model-X", "OEM-A")
        assert isinstance(cl, Checklist)
        assert cl.vehicle_model == "Model-X"
        assert cl.manufacturer == "OEM-A"
        assert cl.total > 0

    def test_checklist_item_ids_unique(self):
        cl = generate_checklist("Model-X", "OEM-A")
        ids = [i.item_id for i in cl.items]
        assert len(ids) == len(set(ids))

    def test_filtered_checklist(self):
        cl_all = generate_checklist("Model-X", "OEM-A")
        cl_cs = generate_checklist(
            "Model-X", "OEM-A", categories=[ClauseCategory.CYBERSECURITY]
        )
        assert cl_cs.total < cl_all.total
        for item in cl_cs.items:
            assert item.category == ClauseCategory.CYBERSECURITY.value

    def test_all_items_start_pending(self):
        cl = generate_checklist("Model-X", "OEM-A")
        for item in cl.items:
            assert item.status == CheckItemStatus.PENDING

    def test_mark_compliant(self):
        cl = generate_checklist("Model-X", "OEM-A")
        item = cl.items[0]
        item.mark_compliant(reviewer="tester", notes="OK")
        assert item.status == CheckItemStatus.COMPLIANT
        assert item.reviewed_by == "tester"
        assert item.reviewed_at is not None

    def test_mark_non_compliant(self):
        cl = generate_checklist("Model-X", "OEM-A")
        item = cl.items[0]
        item.mark_non_compliant(reviewer="tester", notes="Missing data")
        assert item.status == CheckItemStatus.NON_COMPLIANT

    def test_waive_item(self):
        cl = generate_checklist("Model-X", "OEM-A")
        item = cl.items[0]
        item.waive(reason="Not applicable for this variant", reviewer="approver")
        assert item.status == CheckItemStatus.WAIVED
        assert "WAIVED" in item.notes

    def test_summary_counts(self):
        cl = generate_checklist("Model-X", "OEM-A")
        cl.items[0].mark_compliant("r")
        cl.items[1].mark_non_compliant("r")
        cl.items[2].mark_not_applicable()
        s = cl.summary()
        assert s["compliant"] == 1
        assert s["non_compliant"] == 1
        assert s["not_applicable"] == 1

    def test_completion_percent(self):
        cl = generate_checklist("Model-X", "OEM-A")
        for item in cl.items:
            item.mark_compliant("r")
        assert cl.completion_percent == 100.0

    def test_to_json(self):
        import json
        cl = generate_checklist("Model-X", "OEM-A")
        data = json.loads(cl.to_json())
        assert "items" in data
        assert "summary" in data

    def test_items_by_severity(self):
        cl = generate_checklist("Model-X", "OEM-A")
        critical = cl.items_by_severity(Severity.CRITICAL)
        assert all(i.severity == Severity.CRITICAL.value for i in critical)


# ---------------------------------------------------------------------------
# 3. Document validation tests
# ---------------------------------------------------------------------------

class TestDocumentValidation:
    def test_complete_doc_passes(self):
        report = validate_document(
            COMPLETE_DOCUMENT,
            vehicle_model="Model-X",
            manufacturer="OEM-A",
        )
        assert isinstance(report, ValidationReport)
        assert report.passed is True

    def test_incomplete_doc_fails(self):
        report = validate_document(
            INCOMPLETE_DOCUMENT,
            vehicle_model="Model-Y",
            manufacturer="OEM-B",
        )
        assert report.passed is False

    def test_ml_risk_low_for_complete(self):
        report = validate_document(COMPLETE_DOCUMENT, use_ml=True)
        assert report.ml_risk_label == "LOW"
        assert report.ml_risk_score < 0.25

    def test_ml_risk_high_for_incomplete(self):
        report = validate_document(INCOMPLETE_DOCUMENT, use_ml=True)
        assert report.ml_risk_label in ("MEDIUM", "HIGH")

    def test_ml_disabled(self):
        report = validate_document(COMPLETE_DOCUMENT, use_ml=False)
        assert report.ml_risk_score is None
        assert report.ml_risk_label is None

    def test_critical_failures_identified(self):
        report = validate_document(INCOMPLETE_DOCUMENT)
        assert len(report.critical_failures) > 0
        for r in report.critical_failures:
            assert r.severity == Severity.CRITICAL.value
            assert not r.passed

    def test_report_to_json(self):
        import json
        report = validate_document(COMPLETE_DOCUMENT)
        data = json.loads(report.to_json())
        assert "overall_pass" in data
        assert "requirement_results" in data

    def test_invalid_asil_level(self):
        doc = {**COMPLETE_DOCUMENT, "asil_level": "INVALID"}
        report = validate_document(doc)
        asil_result = next(
            (r for r in report.results if r.req_id == "6.2-R1"), None
        )
        assert asil_result is not None
        assert not asil_result.passed

    def test_low_test_coverage_fails(self):
        doc = {**COMPLETE_DOCUMENT, "test_coverage_percent": 50}
        report = validate_document(doc)
        cov_result = next(
            (r for r in report.results if r.req_id == "7.2-R1"), None
        )
        assert cov_result is not None
        assert not cov_result.passed

    def test_invalid_date_format_fails(self):
        doc = {**COMPLETE_DOCUMENT, "homologation_date": "15/11/2024"}
        report = validate_document(doc)
        date_result = next(
            (r for r in report.results if r.req_id == "4.2-R2"), None
        )
        assert date_result is not None
        assert not date_result.passed

    def test_incomplete_tara_fails(self):
        doc = {**COMPLETE_DOCUMENT, "tara": {"threat_list": ["x"]}}
        report = validate_document(doc)
        tara_result = next(
            (r for r in report.results if r.req_id == "5.2-R1"), None
        )
        assert tara_result is not None
        assert not tara_result.passed


# ---------------------------------------------------------------------------
# 4. Compliance Engine (end-to-end) tests
# ---------------------------------------------------------------------------

class TestComplianceEngine:
    def test_create_submission(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        assert sub.submission_id.startswith("SUB-")
        assert sub.status == SubmissionStatus.DRAFT

    def test_generate_checklist(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        cl = engine.generate_checklist(sub.submission_id)
        assert sub.checklist is cl
        assert cl.total > 0

    def test_validate_document_attaches_report(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        report = engine.validate_document(
            sub.submission_id, COMPLETE_DOCUMENT, document_id="DOC-001"
        )
        assert sub.validation_report is report
        assert sub.status == SubmissionStatus.UNDER_REVIEW

    def test_approve_submission(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        engine.validate_document(sub.submission_id, COMPLETE_DOCUMENT)
        engine.approve_submission(sub.submission_id, actor="approver")
        assert sub.status == SubmissionStatus.APPROVED

    def test_reject_submission(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        engine.validate_document(sub.submission_id, INCOMPLETE_DOCUMENT)
        engine.reject_submission(
            sub.submission_id,
            actor="approver",
            reason="Critical failures",
        )
        assert sub.status == SubmissionStatus.REJECTED
        assert "Critical failures" in sub.notes

    def test_review_checklist_item(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        cl = engine.generate_checklist(sub.submission_id)
        item = cl.items[0]
        engine.review_checklist_item(
            sub.submission_id,
            item.item_id,
            CheckItemStatus.COMPLIANT,
            reviewer="rev1",
            notes="Verified",
        )
        assert item.status == CheckItemStatus.COMPLIANT

    def test_review_non_existent_item_raises(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        engine.generate_checklist(sub.submission_id)
        with pytest.raises(ValueError, match="not found"):
            engine.review_checklist_item(
                sub.submission_id, "DOES-NOT-EXIST",
                CheckItemStatus.COMPLIANT, reviewer="r",
            )

    def test_validate_without_checklist_ok(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1")
        # Can validate without having generated a checklist first
        report = engine.validate_document(sub.submission_id, COMPLETE_DOCUMENT)
        assert report is not None

    def test_generate_checklist_without_submission_raises(self, engine):
        with pytest.raises(ValueError):
            engine.generate_checklist("NONEXISTENT-ID")

    def test_dashboard(self, engine):
        engine.create_submission("M1", "OEM", "M1")
        engine.create_submission("M2", "OEM", "M1")
        dashboard = engine.get_compliance_dashboard()
        assert dashboard["total_submissions"] == 2

    def test_audit_events_logged(self, engine):
        sub = engine.create_submission("Model-Z", "OEM-C", "M1", actor="user1")
        engine.validate_document(
            sub.submission_id, COMPLETE_DOCUMENT, actor="validator1"
        )
        engine.approve_submission(sub.submission_id, actor="approver1")
        trail = engine.export_audit_trail()
        event_types = [e["event_type"] for e in trail["events"]]
        assert "SUBMISSION_CREATED" in event_types
        assert "DOCUMENT_VALIDATED" in event_types
        assert "SUBMISSION_APPROVED" in event_types


# ---------------------------------------------------------------------------
# 5. Audit Trail tests
# ---------------------------------------------------------------------------

class TestAuditTrail:
    def test_log_event(self):
        trail = AuditTrail()
        ev = trail.log(
            AuditEventType.SUBMISSION_CREATED,
            actor="admin",
            entity_id="SUB-001",
            entity_type="Submission",
            details={"vehicle_model": "X"},
        )
        assert ev.event_type == AuditEventType.SUBMISSION_CREATED
        assert ev.actor == "admin"
        assert len(trail.events) == 1

    def test_filter_by_entity(self):
        trail = AuditTrail()
        trail.log(AuditEventType.SUBMISSION_CREATED, "a", "SUB-001", "S")
        trail.log(AuditEventType.DOCUMENT_VALIDATED, "a", "SUB-002", "S")
        assert len(trail.events_for_entity("SUB-001")) == 1
        assert len(trail.events_for_entity("SUB-002")) == 1

    def test_filter_by_actor(self):
        trail = AuditTrail()
        trail.log(AuditEventType.SUBMISSION_CREATED, "alice", "S1", "S")
        trail.log(AuditEventType.DOCUMENT_VALIDATED, "bob", "S2", "S")
        assert len(trail.events_by_actor("alice")) == 1
        assert len(trail.events_by_actor("bob")) == 1

    def test_to_json(self):
        import json
        trail = AuditTrail()
        trail.log(AuditEventType.SUBMISSION_CREATED, "a", "S1", "S")
        data = json.loads(trail.to_json())
        assert "events" in data
        assert len(data["events"]) == 1

    def test_events_append_only(self):
        trail = AuditTrail()
        trail.log(AuditEventType.SUBMISSION_CREATED, "a", "S1", "S")
        events_snapshot = trail.events
        events_snapshot.clear()  # Modifying the copy must not affect the trail
        assert len(trail.events) == 1
