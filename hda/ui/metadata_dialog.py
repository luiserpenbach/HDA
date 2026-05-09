"""Auto-generated metadata completion form.

Renders an editable field for each MetadataField the campaign's schema
declares. Pre-fills any value already resolved from a sidecar / campaign
default. On Accept returns a dict of operator-supplied values; the caller
hands it to ``complete_metadata_and_analyze``.

The form fails-loud on coercion errors (so ints stay ints, choices stay
choices) and refuses Accept while required fields are blank — surface
the exact field that's missing inline rather than the analysis service
raising later.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hda.domain.metadata import FieldType, MetadataField, MetadataSchema


class MetadataCompletionDialog(QDialog):
    """Modal that collects values for the metadata schema's required + optional
    fields. ``values()`` returns the operator-supplied dict on accept."""

    def __init__(
        self,
        schema: MetadataSchema,
        existing: Optional[Mapping[str, Any]] = None,
        missing_required: tuple[str, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Complete metadata")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._schema = schema
        self._existing = dict(existing or {})
        self._missing_required = set(missing_required)
        self._editors: dict[str, QWidget] = {}
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color:#7f1d1d; padding:4px;")
        self._error_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "Required fields are marked with *.  Values you supplied "
            "previously (sidecar / campaign defaults) are pre-filled."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#52525b;")
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addLayout(form)

        for f in schema.fields:
            editor = self._build_editor(f)
            self._editors[f.name] = editor
            label_text = self._label_for(f)
            form.addRow(label_text, editor)

        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("Save & analyze")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Focus the first missing-required field if any.
        for f in schema.fields:
            if f.name in self._missing_required:
                self._editors[f.name].setFocus()
                break

    # -------------------------------------------------------------- helpers

    def values(self) -> dict[str, Any]:
        """Return the operator-supplied subset (skips empty optional fields)."""
        out: dict[str, Any] = {}
        for f in self._schema.fields:
            editor = self._editors[f.name]
            v = self._read_editor(f, editor)
            if v is None or v == "":
                continue
            out[f.name] = v
        return out

    def _label_for(self, f: MetadataField) -> str:
        unit = f" [{f.unit}]" if f.unit else ""
        marker = " *" if f.required else ""
        return f"{f.name}{unit}{marker}"

    def _build_editor(self, f: MetadataField) -> QWidget:
        prev = self._existing.get(f.name)
        if f.type is FieldType.CHOICE:
            cb = QComboBox()
            cb.addItem("")  # "no selection"
            cb.addItems(list(f.choices))
            if prev is not None and str(prev) in f.choices:
                cb.setCurrentText(str(prev))
            return cb
        if f.type is FieldType.BOOL:
            ck = QCheckBox()
            ck.setChecked(bool(prev) if prev is not None else False)
            return ck
        if f.type is FieldType.INT:
            sp = QSpinBox()
            sp.setRange(-(10**9), 10**9)
            sp.setSpecialValueText("")  # treat min as "blank"
            try:
                sp.setValue(int(prev) if prev is not None else sp.minimum())
            except (TypeError, ValueError):
                sp.setValue(sp.minimum())
            return sp
        if f.type is FieldType.FLOAT:
            sp = QDoubleSpinBox()
            sp.setDecimals(6)
            sp.setRange(-1e12, 1e12)
            try:
                sp.setValue(float(prev) if prev is not None else 0.0)
            except (TypeError, ValueError):
                sp.setValue(0.0)
            return sp
        # STRING
        le = QLineEdit()
        if prev is not None:
            le.setText(str(prev))
        if f.help:
            le.setPlaceholderText(f.help)
        return le

    def _read_editor(self, f: MetadataField, editor: QWidget) -> Any:
        if isinstance(editor, QComboBox):
            return editor.currentText()
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QSpinBox):
            v = editor.value()
            return None if v == editor.minimum() else v
        if isinstance(editor, QDoubleSpinBox):
            return editor.value()
        if isinstance(editor, QLineEdit):
            return editor.text().strip()
        return None

    def _on_accept(self) -> None:
        values = self.values()
        result = self._schema.validate(values)
        if result.errors:
            msg = "; ".join(f"{e.field_name}: {e.message}" for e in result.errors)
            self._show_error(msg)
            return
        # Required fields not yet supplied here are still missing UNLESS
        # they were already pre-filled in self._existing.
        merged = {**self._existing, **values}
        merged_result = self._schema.validate(merged)
        if merged_result.missing_required:
            self._show_error(
                "Required fields still missing: "
                + ", ".join(merged_result.missing_required)
            )
            return
        self.accept()

    def _show_error(self, text: str) -> None:
        self._error_label.setText(text)
        self._error_label.setVisible(True)
