from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from scipy.optimize import brentq

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
QUERY_CONFIG = BASE_DIR / "config" / "estat_queries.yaml"
ESTAT_API_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"


class DataProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeriesProvenance:
    source: str
    identifier: str
    retrieved_at_utc: str
    query: dict[str, str]
    first_period: str
    last_period: str
    observations: int
    content_sha256: str
    data_kind: str = "observed"


@dataclass(frozen=True)
class CohortResult:
    industry: str
    start_year: int
    horizon_months: int
    first_month: str
    last_month: str
    monthly_irr: float
    annualized_irr: float
    total_contributions: float
    terminal_value: float
    inference_status: str = "descriptive_only_dependent_overlapping_cohorts"


def _directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value is None else [value])


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataProvenanceError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DataProvenanceError(f"{name} is not finite")
    return number


def _year(period: Any) -> int:
    digits = "".join(char for char in str(period) if char.isdigit())
    if len(digits) < 4:
        raise DataProvenanceError(f"Cannot extract year from {period!r}")
    year = int(digits[:4])
    if not 1900 <= year <= 2200:
        raise DataProvenanceError(f"Invalid year: {year}")
    return year


def load_query_config(path: Path = QUERY_CONFIG) -> dict[str, Any]:
    if not path.exists():
        raise DataProvenanceError(f"Missing query config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise DataProvenanceError("Query config must be a mapping")
    if config.get("verified") is not True:
        raise DataProvenanceError(
            "e-Stat dimensions are quarantined. Verify table metadata, industry, "
            "company size, unit, and annual aggregation before setting verified=true."
        )
    if not isinstance(config.get("queries"), list) or not config["queries"]:
        raise DataProvenanceError("No verified e-Stat queries configured")
    return config


def fetch_estat_series(
    *,
    app_id: str,
    stats_data_id: str,
    filters: Mapping[str, Any],
    expected_unit: str,
    session: requests.Session | None = None,
) -> tuple[dict[int, float], SeriesProvenance]:
    if not app_id:
        raise DataProvenanceError("ESTAT_APP_ID is required")
    if not stats_data_id or not filters or not expected_unit:
        raise DataProvenanceError("stats_data_id, filters, and expected_unit are required")

    params = {
        "appId": app_id,
        "statsDataId": str(stats_data_id),
        "metaGetFlg": "Y",
        "cntGetFlg": "N",
        "explanationGetFlg": "N",
        "annotationGetFlg": "Y",
        **{str(key): str(value) for key, value in filters.items()},
    }
    response = (session or requests.Session()).get(
        ESTAT_API_URL, params=params, timeout=60
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise DataProvenanceError("e-Stat did not return JSON") from exc

    root = payload.get("GET_STATS_DATA", {})
    result = root.get("RESULT", {})
    if str(result.get("STATUS", "")) not in {"0", "00"}:
        raise DataProvenanceError(
            f"e-Stat status={result.get('STATUS')}: {result.get('ERROR_MSG')}"
        )
    values = _as_list(
        root.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE")
    )
    if not values:
        raise DataProvenanceError("e-Stat returned no observations")

    units = {
        str(row.get("@unit"))
        for row in values
        if isinstance(row, Mapping) and row.get("@unit") is not None
    }
    if units and units != {str(expected_unit)}:
        raise DataProvenanceError(
            f"Unexpected units={sorted(units)}; expected={expected_unit!r}"
        )

    rows_by_year: dict[int, list[Mapping[str, Any]]] = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise DataProvenanceError("Invalid VALUE record")
        rows_by_year.setdefault(_year(row.get("@time")), []).append(row)
    duplicates = {year: rows for year, rows in rows_by_year.items() if len(rows) != 1}
    if duplicates:
        remaining = sorted(
            {
                str(key)
                for rows in duplicates.values()
                for row in rows
                for key in row
                if str(key).startswith("@") and key not in {"@time", "@unit"}
            }
        )
        raise DataProvenanceError(
            "Filters did not yield one observation per year; "
            f"duplicate_years={sorted(duplicates)}, remaining_dimensions={remaining}"
        )

    series = {
        year: _finite(rows[0].get("$"), f"e-Stat value {year}")
        for year, rows in rows_by_year.items()
    }
    if any(value < 0 for value in series.values()):
        raise DataProvenanceError("Negative wage or bonus observation")
    periods = sorted(series)
    digest = hashlib.sha256(response.content).hexdigest()
    return series, SeriesProvenance(
        source="e-Stat API",
        identifier=str(stats_data_id),
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        query={key: value for key, value in params.items() if key != "appId"},
        first_period=str(periods[0]),
        last_period=str(periods[-1]),
        observations=len(periods),
        content_sha256=digest,
    )


def fetch_wage_bonus(
    app_id: str, config_path: Path = QUERY_CONFIG
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, SeriesProvenance]]]:
    config = load_query_config(config_path)
    data: dict[str, dict[str, dict[int, float]]] = {}
    provenance: dict[str, dict[str, SeriesProvenance]] = {}
    for entry in config["queries"]:
        if not isinstance(entry, Mapping):
            raise DataProvenanceError("Each query must be a mapping")
        industry = str(entry.get("industry", ""))
        if not industry or industry in data:
            raise DataProvenanceError(f"Invalid or duplicate industry: {industry!r}")
        data[industry], provenance[industry] = {}, {}
        for name in ("wage", "bonus"):
            spec = entry.get(name)
            if not isinstance(spec, Mapping):
                raise DataProvenanceError(f"Missing {industry}.{name} query")
            values, source = fetch_estat_series(
                app_id=app_id,
                stats_data_id=str(spec.get("stats_data_id", "")),
                filters=spec.get("filters", {}),
                expected_unit=str(spec.get("expected_unit", "")),
            )
            data[industry][name] = values
            provenance[industry][name] = source
    return data, provenance


def fetch_world_returns(
    ticker: str = "ACWI", start: str = "2004-01-01", end: str | None = None
) -> tuple[pd.Series, SeriesProvenance]:
    downloaded = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if downloaded.empty or "Close" not in downloaded:
        raise DataProvenanceError(f"No Close data returned for {ticker}")
    close = downloaded["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise DataProvenanceError("Multiple Close series returned")
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if close.index.has_duplicates or len(close) < 13:
        raise DataProvenanceError("Market prices are duplicated or insufficient")
    if (close <= 0).any() or not np.isfinite(close.to_numpy()).all():
        raise DataProvenanceError("Market prices must be finite and positive")
    returns = close.pct_change(fill_method=None).dropna()
    if (returns <= -1).any() or not np.isfinite(returns.to_numpy()).all():
        raise DataProvenanceError("Invalid market return")
    csv_bytes = returns.rename("return").to_csv().encode("utf-8")
    return returns, SeriesProvenance(
        source="Yahoo Finance via yfinance",
        identifier=ticker,
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        query={"ticker": ticker, "start": start, "end": end or "latest", "interval": "1mo"},
        first_period=returns.index.min().isoformat(),
        last_period=returns.index.max().isoformat(),
        observations=len(returns),
        content_sha256=hashlib.sha256(csv_bytes).hexdigest(),
    )


def periodic_irr(cashflows: Sequence[float]) -> float:
    flows = np.asarray(cashflows, dtype=float)
    if flows.ndim != 1 or flows.size < 2 or not np.isfinite(flows).all():
        raise ValueError("cashflows must contain at least two finite values")
    signs = np.sign(flows[flows != 0])
    if not np.any(signs < 0) or not np.any(signs > 0):
        raise ValueError("cashflows need both contributions and receipt")
    if int(np.sum(signs[1:] != signs[:-1])) != 1:
        raise ValueError("IRR may not be unique with multiple sign changes")

    def npv(rate: float) -> float:
        return float(np.sum(flows / np.power(1 + rate, np.arange(flows.size))))

    lower, upper = -0.999999, 1.0
    for _ in range(60):
        left, right = npv(lower), npv(upper)
        if left * right < 0:
            return float(brentq(npv, lower, upper, xtol=1e-12, rtol=1e-12))
        upper = upper * 2 + 1
    raise ValueError("Unable to bracket a unique IRR root")


def annualize_monthly_irr(monthly_irr: float) -> float:
    if monthly_irr <= -1 or not math.isfinite(monthly_irr):
        raise ValueError("monthly_irr must be finite and greater than -100%")
    return (1 + monthly_irr) ** 12 - 1


def calculate_cohort(
    industry: str,
    wage: Mapping[int, float],
    bonus: Mapping[int, float],
    returns: pd.Series,
    start_year: int,
    horizon_months: int,
) -> CohortResult:
    selected = returns[returns.index.year >= start_year].iloc[:horizon_months]
    if horizon_months < 2 or len(selected) != horizon_months:
        raise DataProvenanceError("Incomplete fixed-horizon market history")
    portfolio, contributions = 0.0, 0.0
    cashflows: list[float] = []
    for timestamp, monthly_return in selected.items():
        year = int(timestamp.year)
        if year not in wage or year not in bonus:
            raise DataProvenanceError(f"Missing wage or bonus for {year}")
        contribution = _finite(wage[year], "wage") + _finite(bonus[year], "bonus") / 12
        if contribution <= 0:
            raise DataProvenanceError(f"Non-positive contribution for {year}")
        portfolio = (portfolio + contribution) * (1 + float(monthly_return))
        contributions += contribution
        cashflows.append(-contribution)
    cashflows.append(portfolio)
    monthly = periodic_irr(cashflows)
    return CohortResult(
        industry=industry,
        start_year=start_year,
        horizon_months=horizon_months,
        first_month=selected.index.min().strftime("%Y-%m"),
        last_month=selected.index.max().strftime("%Y-%m"),
        monthly_irr=monthly,
        annualized_irr=annualize_monthly_irr(monthly),
        total_contributions=contributions,
        terminal_value=portfolio,
    )


def build_results(
    data: Mapping[str, Mapping[str, Mapping[int, float]]],
    returns: pd.Series,
    horizon_months: int,
) -> list[CohortResult]:
    results: list[CohortResult] = []
    for industry, series in data.items():
        for start_year in sorted(set(series["wage"]) & set(series["bonus"])):
            try:
                results.append(
                    calculate_cohort(
                        industry,
                        series["wage"],
                        series["bonus"],
                        returns,
                        start_year,
                        horizon_months,
                    )
                )
            except DataProvenanceError:
                continue
    if not results:
        raise DataProvenanceError("No complete fixed-horizon cohorts")
    return results


def descriptive_summary(results: Sequence[CohortResult]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(asdict(item) for item in results)
    output = []
    for industry, group in frame.groupby("industry", sort=True):
        values = group["annualized_irr"].to_numpy(float)
        output.append(
            {
                "industry": industry,
                "cohorts": len(values),
                "mean_annualized_irr": float(values.mean()),
                "median_annualized_irr": float(np.median(values)),
                "min_annualized_irr": float(values.min()),
                "max_annualized_irr": float(values.max()),
                "inference_status": "not_tested",
                "reason": "Overlapping cohorts share the same market path; ordinary independent-sample tests and t confidence intervals are invalid.",
            }
        )
    return output


def write_outputs(
    results: Sequence[CohortResult],
    summaries: Sequence[Mapping[str, Any]],
    estat_provenance: Mapping[str, Mapping[str, SeriesProvenance]],
    market_provenance: SeriesProvenance,
) -> None:
    _directories()
    rows = [asdict(item) for item in results]
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "irr_results.csv", index=False)
    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(list(summaries), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": asdict(market_provenance),
        "estat": {
            industry: {name: asdict(item) for name, item in sources.items()}
            for industry, sources in estat_provenance.items()
        },
        "synthetic_fallback_used": False,
        "inferential_statistics_generated": False,
    }
    (PROCESSED_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result_table = pd.DataFrame(rows).to_html(index=False, border=0)
    summary_table = pd.DataFrame(summaries).to_html(index=False, border=0)
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>賃金積立IRR</title><style>body{{font-family:system-ui;max-width:1200px;margin:auto;padding:2rem}}table{{border-collapse:collapse;display:block;overflow:auto}}th,td{{border:1px solid #ccc;padding:.4rem}}.warning{{border:2px solid #933;padding:1rem;background:#fff4f4}}</style></head><body><h1>賃金・賞与積立IRR</h1><div class="warning">推論統計は生成していません。開始年コホートは期間が重複し、同一の市場系列を共有します。比較期間は固定しています。</div><h2>結果</h2>{result_table}<h2>記述統計</h2>{summary_table}<p>API失敗時の合成データ代替はありません。出典クエリとハッシュは provenance.json に保存します。</p></body></html>"""
    (REPORTS_DIR / "irr_analysis_report.html").write_text(html, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-config", type=Path, default=QUERY_CONFIG)
    parser.add_argument("--market-ticker", default="ACWI")
    parser.add_argument("--market-start", default="2004-01-01")
    parser.add_argument("--market-end", default=None)
    parser.add_argument("--horizon-months", type=int, default=60)
    args = parser.parse_args(argv)
    _directories()
    data, estat_sources = fetch_wage_bonus(
        os.getenv("ESTAT_APP_ID", ""), args.query_config
    )
    returns, market_source = fetch_world_returns(
        args.market_ticker, args.market_start, args.market_end
    )
    results = build_results(data, returns, args.horizon_months)
    summaries = descriptive_summary(results)
    write_outputs(results, summaries, estat_sources, market_source)
    print(json.dumps({"cohorts": len(results), "synthetic_fallback_used": False, "inferential_statistics_generated": False}))


if __name__ == "__main__":
    main()
