"""JSON schema and Pydantic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError

from schemas.case import FraudCase
from utils.config import PROJECT_ROOT


_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "case_schema.json"


def _load_json_schema() -> dict[str, Any]:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_case_schema(case_dict: dict[str, Any]) -> list[str]:
    """Return list of schema errors (empty if valid)."""
    schema = _load_json_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(case_dict), key=lambda e: e.path)
    return [f"{list(e.path)}: {e.message}" for e in errors]


def validate_case_dict(case_dict: dict[str, Any]) -> tuple[FraudCase | None, list[str], list[str]]:
    """
    Validate and parse case.
    Returns (fraud_case, schema_errors, pydantic_errors).
    """
    schema_errors = validate_case_schema(case_dict)
    if schema_errors:
        return None, schema_errors, []

    try:
        case = FraudCase.model_validate(case_dict)
        return case, [], []
    except ValidationError as e:
        return None, [], [str(e)]
