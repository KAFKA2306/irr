from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "estat_queries.yaml"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INVALID_TOKENS = {"", "*", "all", "any", "unknown", "unverified", "todo", "tbd", "none", "null"}
_REQUIRED_SPEC_FIELDS = {
    "stats_data_id",
    "table_title",
    "source_url",
    "retrieved_at",
    "metadata_sha256",
    "population_scope",
    "company_size_label",
    "sex_label",
    "age_label",
    "measure_label",
    "expected_unit",
    "frequency",
    "filters",
    "expected_dimensions",
}


class ContractError(ValueError):
    pass


def _invalid(value: Any) -> bool:
    return str(value).strip().lower() in _INVALID_TOKENS


def _iso_date(value: Any) -> bool:
    try:
        dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _validate_selector_mapping(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not value:
        errors.append(f"{path} must be a non-empty mapping")
        return
    for key, selected in value.items():
        if _invalid(key) or _invalid(selected):
            errors.append(f"{path}.{key} contains an empty, wildcard, or placeholder selector")


def validate_contract(config: Mapping[str, Any], *, require_verified: bool = True) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if require_verified and config.get("verified") is not True:
        errors.append("verified must be true")

    queries = config.get("queries")
    if config.get("verified") is not True:
        if queries not in ([], None):
            errors.append("unverified configuration must not contain executable queries")
        if not str(config.get("quarantine_reason", "")).strip():
            errors.append("unverified configuration requires quarantine_reason")
        return errors

    if not isinstance(queries, list) or not queries:
        errors.append("verified configuration requires at least one query")
        return errors

    seen_industries: set[str] = set()
    for index, entry in enumerate(queries):
        prefix = f"queries[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{prefix} must be a mapping")
            continue
        industry = str(entry.get("industry", "")).strip()
        if _invalid(industry):
            errors.append(f"{prefix}.industry is required")
        elif industry in seen_industries:
            errors.append(f"duplicate industry: {industry}")
        seen_industries.add(industry)

        for kind in ("wage", "bonus"):
            spec_path = f"{prefix}.{kind}"
            spec = entry.get(kind)
            if not isinstance(spec, Mapping):
                errors.append(f"{spec_path} must be a mapping")
                continue
            missing = sorted(_REQUIRED_SPEC_FIELDS - set(spec))
            if missing:
                errors.append(f"{spec_path} missing fields: {', '.join(missing)}")
                continue
            stats_data_id = str(spec["stats_data_id"]).strip()
            if not stats_data_id.isdigit():
                errors.append(f"{spec_path}.stats_data_id must contain only digits")
            for field in (
                "table_title",
                "population_scope",
                "company_size_label",
                "sex_label",
                "age_label",
                "measure_label",
                "expected_unit",
            ):
                if _invalid(spec[field]):
                    errors.append(f"{spec_path}.{field} must be explicit")
            if str(spec["frequency"]).lower() != "annual":
                errors.append(f"{spec_path}.frequency must be annual")
            if not str(spec["source_url"]).startswith("https://www.e-stat.go.jp/"):
                errors.append(f"{spec_path}.source_url must be an official e-Stat https URL")
            if not _iso_date(spec["retrieved_at"]):
                errors.append(f"{spec_path}.retrieved_at must be an ISO date")
            digest = str(spec["metadata_sha256"]).lower()
            if not _SHA256_RE.fullmatch(digest):
                errors.append(f"{spec_path}.metadata_sha256 must be a SHA-256 digest")
            _validate_selector_mapping(spec["filters"], f"{spec_path}.filters", errors)
            _validate_selector_mapping(
                spec["expected_dimensions"], f"{spec_path}.expected_dimensions", errors
            )
    return errors


def load_and_validate(
    path: Path = DEFAULT_CONFIG, *, require_verified: bool = True
) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ContractError("e-Stat contract must be a mapping")
    errors = validate_contract(raw, require_verified=require_verified)
    if errors:
        raise ContractError("; ".join(errors))
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the e-Stat evidence contract")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-quarantined",
        action="store_true",
        help="accept a structurally valid verified=false quarantine file",
    )
    args = parser.parse_args()
    try:
        contract = load_and_validate(
            args.config, require_verified=not args.allow_quarantined
        )
    except (OSError, ContractError) as exc:
        print(f"e-Stat contract invalid: {exc}")
        return 1
    state = "verified" if contract.get("verified") is True else "quarantined"
    print(f"e-Stat contract state: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
