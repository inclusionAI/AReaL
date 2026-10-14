"""W&B and Prometheus adapters for live and fixture collection."""

import json
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from autopilot_harness.contracts.monitor import SourceBatch, SourceReport

from ._io import (
    SourceError,
    base_url,
    endpoint_reference,
    positive_int,
    positive_number,
    read_json,
    resolve_path,
    timestamp,
)
from .outputs import sample

# W&B: recorded configuration and sparse metric history.


def wandb_run_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("wandb.run must be a string")
    if "://" in value:
        parsed = urlsplit(value)
        base_url(f"{parsed.scheme}://{parsed.netloc}")
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[2] != "runs":
            raise ValueError("W&B URL must contain /entity/project/runs/run_id")
        parts.pop(2)
    else:
        parts = value.strip("/").split("/")
    if len(parts) != 3 or not all(part and part not in {".", ".."} for part in parts):
        raise ValueError("W&B run must be entity/project/run_id or a run URL")
    return "/".join(parts)


def validate_wandb(options: dict) -> None:
    wandb_run_path(options["run"])
    if not isinstance(options.get("metrics"), dict) or not options["metrics"]:
        raise ValueError(
            "wandb.metrics must map selected metric names to units (or null)"
        )
    if any(
        not isinstance(key, str)
        or not key
        or (unit is not None and not isinstance(unit, str))
        for key, unit in options["metrics"].items()
    ):
        raise ValueError(
            "W&B metric names and units must be strings; unknown units may be null"
        )
    positive_int(options.get("page_size", 1000), "wandb.page_size")
    positive_int(options.get("max_rows", 100000), "wandb.max_rows")
    positive_number(options.get("timeout_seconds", 30), "wandb.timeout_seconds")
    if options.get("base_url"):
        base_url(options["base_url"])


def _wandb_api(options: dict):
    key = os.environ.get(options.get("api_key_env", "WANDB_API_KEY"), "")
    if not key:
        key = os.environ.get("ALAB_WANDB_API_KEY", "")
    if not key:
        raise SourceError("W&B key missing: set WANDB_API_KEY or ALAB_WANDB_API_KEY")
    try:
        import wandb
    except ImportError as exc:
        raise SourceError("W&B SDK missing: install autopilot-harness[wandb]") from exc
    endpoint = (
        options.get("base_url")
        or os.environ.get("WANDB_BASE_URL")
        or os.environ.get("ALAB_WANDB_INTERNAL_BASE_URL")
    )
    if endpoint is None and "://" in options["run"]:
        host = urlsplit(options["run"])
        if host.hostname not in {"wandb.ai", "www.wandb.ai"}:
            raise SourceError(
                "private W&B requires explicit base_url or WANDB_BASE_URL"
            )
    endpoint = base_url(endpoint or "https://api.wandb.ai")
    return wandb.Api(
        api_key=key,
        timeout=options.get("timeout_seconds", 30),
        overrides={"base_url": endpoint},
    )


def collect_wandb(options: dict, root: Path, *, api_factory=None) -> SourceBatch:
    validate_wandb(options)
    reference = wandb_run_path(options["run"])
    mode = "fixture" if options.get("fixture") else "live"
    report = SourceReport("wandb", mode, "ok", reference)
    batch = SourceBatch([], report)
    rows_read = 0
    seen = set()
    try:
        if mode == "fixture":
            data = read_json(resolve_path(options["fixture"], root))
            rows = iter(data["history"])
            batch.recorded_configuration = data.get("config")
        else:
            api = (api_factory or _wandb_api)(options)
            run = api.run(reference)
            batch.recorded_configuration = dict(run.config)
            # Passing keys to scan_history drops sparse rows missing any key.
            rows = iter(run.scan_history(page_size=options.get("page_size", 1000)))
        if batch.recorded_configuration is not None and not isinstance(
            batch.recorded_configuration, dict
        ):
            batch.recorded_configuration = None
            raise SourceError("W&B configuration must be an object")
        for index, row in enumerate(rows):
            if index >= options.get("max_rows", 100000):
                report.status = "partial"
                report.errors.append("history truncated by max_rows")
                break
            if not isinstance(row, dict):
                raise SourceError("W&B history row must be an object")
            rows_read += 1
            for metric, unit in options["metrics"].items():
                if metric not in row:
                    continue
                seen.add(metric)
                batch.observations.append(
                    sample(
                        metric=metric,
                        value=row[metric],
                        unit=unit,
                        entity={"run": reference},
                        observed_at=row.get("_timestamp"),
                        available_at=time.time(),
                        timestamp_basis="wandb_row_timestamp",
                        window={"kind": "logged_row"},
                        coverage={
                            "present": int(row[metric] is not None),
                            "expected": 1,
                        },
                        source_ref=f"wandb:{reference}#row={index}",
                        mode=mode,
                        logging_step=row.get("_step"),
                        train_step=row.get(options.get("train_step_key")),
                    )
                )
    except Exception as exc:
        report.status = (
            "partial"
            if batch.observations or batch.recorded_configuration is not None
            else "error"
        )
        # SDK exceptions may embed request headers or credentials. Never serialize them.
        report.errors.append(
            str(exc)
            if isinstance(exc, SourceError)
            else f"W&B read failed ({type(exc).__name__})"
        )
    missing = sorted(set(options["metrics"]) - seen)
    invalid = sum(item.value is None for item in batch.observations)
    report.details = {
        "rows_read": rows_read,
        "missing_metrics": missing,
        "invalid_samples": invalid,
        "fetched_at": time.time(),
        "history_kind": "scan_history",
        "step_alignment": "explicit_key_only",
    }
    if mode == "fixture":
        report.details["fixture"] = str(resolve_path(options["fixture"], root))
    else:
        report.details["api_base_url"] = endpoint_reference(
            options.get("base_url")
            or os.environ.get("WANDB_BASE_URL")
            or os.environ.get("ALAB_WANDB_INTERNAL_BASE_URL")
            or "https://api.wandb.ai"
        )
    if report.status == "ok":
        report.status = (
            "empty"
            if not batch.observations
            else "partial"
            if missing or invalid
            else "ok"
        )
    return batch


# Prometheus: range queries with series labels and evaluation timestamps.


def validate_prometheus(options: dict) -> None:
    start, end = timestamp(options["start"]), timestamp(options["end"])
    step = positive_number(options["step_seconds"], "prometheus.step_seconds")
    if start > end:
        raise ValueError("prometheus.start must be <= end")
    if math.floor((end - start) / step) + 1 > 11000:
        raise ValueError(
            "Prometheus range exceeds 11000 evaluations per series; increase step_seconds"
        )
    positive_number(options.get("timeout_seconds", 30), "prometheus.timeout_seconds")
    positive_int(
        options.get("max_response_bytes", 16777216), "prometheus.max_response_bytes"
    )
    queries = options.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("prometheus.queries must be a non-empty list")
    names = set()
    for query in queries:
        if (
            not isinstance(query, dict)
            or not query.get("name")
            or not query.get("expression")
        ):
            raise ValueError("each Prometheus query requires name and expression")
        if not isinstance(query["name"], str) or not isinstance(
            query["expression"], str
        ):
            raise ValueError("Prometheus query name and expression must be strings")
        if query["name"] in names:
            raise ValueError("Prometheus query names must be unique")
        names.add(query["name"])
        if query.get("unit") is not None and not isinstance(query["unit"], str):
            raise ValueError("Prometheus query unit must be a string or null")
    if options.get("base_url"):
        base_url(options["base_url"])


def query_prometheus_range(options: dict, expression: str) -> dict:
    endpoint = (
        options.get("base_url")
        or os.environ.get("PROMETHEUS_URL")
        or os.environ.get("ALAB_PROMETHEUS_URL")
    )
    if not endpoint:
        raise SourceError(
            "Prometheus address missing: set base_url or PROMETHEUS_URL/ALAB_PROMETHEUS_URL"
        )
    fields = {
        "query": expression,
        "start": timestamp(options["start"]),
        "end": timestamp(options["end"]),
        "step": options["step_seconds"],
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    token = os.environ.get(
        options.get("bearer_token_env", "PROMETHEUS_BEARER_TOKEN"), ""
    )
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(
        base_url(endpoint) + "/api/v1/query_range",
        data=urlencode(fields).encode(),
        headers=headers,
        method="POST",
    )
    limit = options.get("max_response_bytes", 16777216)
    try:
        with urlopen(request, timeout=options.get("timeout_seconds", 30)) as response:
            data = response.read(limit + 1)
    except HTTPError as exc:
        raise SourceError(f"Prometheus HTTP {exc.code}") from exc
    if len(data) > limit:
        raise SourceError("Prometheus response exceeds max_response_bytes")
    return json.loads(data)


def collect_prometheus(options: dict, root: Path, *, transport=None) -> SourceBatch:
    validate_prometheus(options)
    mode = "fixture" if options.get("fixture") else "live"
    start, end = timestamp(options["start"]), timestamp(options["end"])
    step = float(options["step_seconds"])
    report = SourceReport("prometheus", mode, "ok", "query_range")
    batch = SourceBatch([], report)
    reports = []
    fixture = None
    if mode == "fixture":
        try:
            fixture = read_json(resolve_path(options["fixture"], root))
        except Exception as exc:
            report.status = "error"
            report.errors.append(
                f"Prometheus fixture read failed ({type(exc).__name__})"
            )
            return batch
    for query in options["queries"]:
        query_report = {
            "name": query["name"],
            "expression": query["expression"],
            "status": "ok",
            "series": 0,
            "samples": 0,
            "warnings": [],
        }
        query_samples = []
        try:
            payload = (
                fixture[query["name"]]
                if fixture is not None
                else (transport or query_prometheus_range)(options, query["expression"])
            )
            if payload.get("status") != "success":
                raise SourceError("Prometheus returned an API error")
            data = payload["data"]
            if data.get("resultType") != "matrix":
                raise SourceError("Prometheus query_range must return a matrix")
            query_report["warnings"] = payload.get("warnings", [])
            query_report["infos"] = payload.get("infos", [])
            for series_index, series in enumerate(data["result"]):
                if series.get("histograms"):
                    raise SourceError(
                        "native histogram samples require an explicit scalar PromQL expression"
                    )
                values = series.get("values", [])
                labels = series["metric"]
                if not isinstance(labels, dict):
                    raise SourceError("Prometheus series labels must be an object")
                coverage = {
                    "present": len(values),
                    "expected": math.floor((end - start) / step) + 1,
                    "basis": "query_evaluations_per_returned_series",
                }
                for sample_index, (observed_at, value) in enumerate(values):
                    if not start <= timestamp(observed_at) <= end:
                        raise SourceError(
                            "Prometheus sample is outside the requested window"
                        )
                    query_samples.append(
                        sample(
                            metric=query["name"],
                            value=value,
                            unit=query.get("unit"),
                            entity=labels,
                            observed_at=observed_at,
                            available_at=time.time(),
                            timestamp_basis="prometheus_query_evaluation",
                            window={
                                "start": start,
                                "end": end,
                                "step_seconds": step,
                                "expression": query["expression"],
                            },
                            coverage=coverage,
                            source_ref=f"prometheus:{query['name']}#series={series_index}&sample={sample_index}",
                            mode=mode,
                        )
                    )
                query_report["series"] += 1
                if len(values) < coverage["expected"]:
                    query_report["status"] = "partial"
            query_report["samples"] = len(query_samples)
            if (
                any(item.value is None for item in query_samples)
                or query_report["warnings"]
            ):
                query_report["status"] = "partial"
            if not query_samples:
                query_report["status"] = "empty"
            batch.observations.extend(query_samples)
        except Exception as exc:
            query_report["status"] = "error"
            error = (
                str(exc)
                if isinstance(exc, SourceError)
                else f"Prometheus read failed ({type(exc).__name__})"
            )
            query_report["error"] = error
            report.errors.append(f"{query['name']}: {error}")
        reports.append(query_report)
    states = {item["status"] for item in reports}
    if states == {"error"}:
        report.status = "error"
    elif states == {"empty"}:
        report.status = "empty"
    elif states != {"ok"}:
        report.status = "partial"
    report.details = {
        "queries": reports,
        "start": start,
        "end": end,
        "step_seconds": step,
        "fetched_at": time.time(),
    }
    if mode == "fixture":
        report.details["fixture"] = str(resolve_path(options["fixture"], root))
    else:
        report.details["api_base_url"] = endpoint_reference(
            options.get("base_url")
            or os.environ.get("PROMETHEUS_URL")
            or os.environ.get("ALAB_PROMETHEUS_URL")
        )
    return batch
