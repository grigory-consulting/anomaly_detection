from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TimestampRange = Tuple[pd.Timestamp, pd.Timestamp]


@dataclass
class LoadedSeries:
    series: pd.Series
    labels: pd.Series
    ranges: List[TimestampRange]
    metadata: Dict[str, Any]


def _coerce_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    missing_before = int(s.isna().sum())
    if missing_before:
        s = s.interpolate(method="time", limit_direction="both")
        s = s.ffill().bfill()
    return s.astype(float), missing_before


def labels_to_ranges(labels: pd.Series) -> List[TimestampRange]:
    if labels.empty:
        return []
    lbl = labels.astype(bool)
    ranges: List[TimestampRange] = []
    in_range = False
    start: Optional[pd.Timestamp] = None
    prev_ts: Optional[pd.Timestamp] = None
    for ts, flag in lbl.items():
        if flag and not in_range:
            start = ts
            in_range = True
        if not flag and in_range and start is not None and prev_ts is not None:
            ranges.append((start, prev_ts))
            in_range = False
            start = None
        prev_ts = ts
    if in_range and start is not None:
        ranges.append((start, lbl.index[-1]))
    return ranges


def ranges_to_point_labels(index: pd.DatetimeIndex, ranges: Sequence[TimestampRange]) -> pd.Series:
    out = pd.Series(False, index=index)
    for start, end in ranges:
        out.loc[start:end] = True
    return out


def windows_json_to_ranges(windows: Iterable[Sequence[str]]) -> List[TimestampRange]:
    ranges: List[TimestampRange] = []
    for item in windows:
        if not item or len(item) != 2:
            continue
        start = pd.Timestamp(item[0])
        end = pd.Timestamp(item[1])
        ranges.append((start, end))
    return sorted(ranges, key=lambda x: x[0])


def deterministic_split_bounds(n: int, train_frac: float = 0.6, val_frac: float = 0.2) -> Tuple[int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 points for train/val/test split, got {n}.")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_end = min(max(train_end, 1), n - 2)
    val_end = min(max(val_end, train_end + 1), n - 1)
    return train_end, val_end


def load_generic_series_csv(path: str | Path) -> LoadedSeries:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    df = pd.read_csv(p)
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError("Expected columns: timestamp,value and optionally label.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")

    value, missing_before = _coerce_series(df["value"])
    labels = pd.Series(False, index=value.index)
    missing_label_count = 0
    if "label" in df.columns:
        raw = pd.to_numeric(df["label"], errors="coerce")
        missing_label_count = int(raw.isna().sum())
        labels = raw.fillna(0).astype(int).astype(bool)
        labels.index = value.index

    ranges = labels_to_ranges(labels)
    return LoadedSeries(
        series=value,
        labels=labels,
        ranges=ranges,
        metadata={
            "source": str(p),
            "missing_value_count": missing_before,
            "missing_label_count": missing_label_count,
            "n_points": int(len(value)),
        },
    )


def load_cloud_prepared_series(path: str | Path) -> LoadedSeries:
    loaded = load_generic_series_csv(path)
    loaded.metadata["dataset"] = "cloud_prepared"
    return loaded


def load_nab_series(
    nab_root: str | Path,
    series_relpath: str,
    labels_path: str | Path = "labels/combined_windows.json",
) -> LoadedSeries:
    root = Path(nab_root)
    data_path = root / "data" / series_relpath
    if not data_path.exists():
        raise FileNotFoundError(f"NAB series not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    series, missing_before = _coerce_series(df["value"])

    lp = Path(labels_path)
    if not lp.is_absolute():
        lp = root / lp
    if not lp.exists():
        raise FileNotFoundError(f"NAB labels file not found: {lp}")

    labels_json = json.loads(lp.read_text())
    windows = labels_json.get(series_relpath, [])
    ranges = windows_json_to_ranges(windows)
    labels = ranges_to_point_labels(series.index, ranges)

    return LoadedSeries(
        series=series,
        labels=labels,
        ranges=ranges,
        metadata={
            "dataset": "nab",
            "source": str(data_path),
            "labels_file": str(lp),
            "series_relpath": series_relpath,
            "n_windows": int(len(ranges)),
            "missing_value_count": missing_before,
            "missing_label_count": 0,
            "n_points": int(len(series)),
        },
    )
