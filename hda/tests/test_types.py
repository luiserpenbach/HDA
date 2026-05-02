"""Domain type invariants."""

from __future__ import annotations

from datetime import datetime

import pytest

from hda.domain import (
    AnalysisResult,
    Hardware,
    MeasurementWithUncertainty,
    Provenance,
    QCFinding,
    QCReport,
    QCStatus,
    SteadyWindow,
    TraceabilityRecord,
)


def test_measurement_rejects_negative_uncertainty():
    with pytest.raises(ValueError):
        MeasurementWithUncertainty(name="cd", value=0.65, uncertainty=-0.01, unit="")


def test_measurement_rel_uncertainty_zero_value():
    m = MeasurementWithUncertainty(name="x", value=0.0, uncertainty=0.1, unit="")
    assert m.rel_uncertainty_pct is None


def test_measurement_rel_uncertainty_negative_value():
    m = MeasurementWithUncertainty(name="dp", value=-2.0, uncertainty=0.1, unit="bar")
    assert m.rel_uncertainty_pct == pytest.approx(5.0)


def test_hardware_requires_both_fields():
    with pytest.raises(ValueError):
        Hardware(part_number="", serial_number="SN-1")
    with pytest.raises(ValueError):
        Hardware(part_number="PN-1", serial_number="")


def test_steady_window_validates_bounds():
    with pytest.raises(ValueError):
        SteadyWindow(start_s=5.0, end_s=5.0, method="cv", confidence=1.0)
    with pytest.raises(ValueError):
        SteadyWindow(start_s=5.0, end_s=4.0, method="cv", confidence=1.0)
    with pytest.raises(ValueError):
        SteadyWindow(start_s=0.0, end_s=1.0, method="cv", confidence=1.5)


def test_qc_report_passed_logic():
    findings = [
        QCFinding("ts_monotonic", QCStatus.PASS, "", blocking=True),
        QCFinding("range", QCStatus.WARN, "near max", blocking=False),
    ]
    assert QCReport(findings=findings).passed is True

    findings_with_block = [
        *findings,
        QCFinding("flatline", QCStatus.FAIL, "PT-01 flat", blocking=True),
    ]
    rep = QCReport(findings=findings_with_block)
    assert rep.passed is False
    assert len(rep.blocking_failures) == 1


def test_analysis_result_measurement_keys_must_match_names():
    m = MeasurementWithUncertainty(
        name="cd", value=0.65, uncertainty=0.01, unit="", provenance=Provenance.SENSOR
    )
    qc = QCReport(findings=[])
    trace = TraceabilityRecord(
        file_hash="a" * 64,
        config_hash="b" * 64,
        metadata_hash="c" * 64,
        processing_version="3.0.0",
        plugin_name="cold_flow",
        plugin_version="1.0.0",
        analyst="op",
        analyzed_at=datetime.utcnow(),
    )
    AnalysisResult(measurements={"cd": m}, qc_report=qc, traceability=trace, confidence=0.9)
    with pytest.raises(ValueError):
        AnalysisResult(
            measurements={"WRONG_KEY": m},
            qc_report=qc,
            traceability=trace,
            confidence=0.9,
        )
