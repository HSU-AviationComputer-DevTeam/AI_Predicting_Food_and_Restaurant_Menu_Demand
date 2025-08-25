#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leaderboard-style sMAPE evaluator.

Definition (provided):
  - Score = average over shops s (weighted if provided, otherwise equal weights)
      of: average over items i in shop s
          of: average over time t in [first 7 days of the TEST file]
              of: 2 * |A_{t,i} - P_{t,i}| / (|A_{t,i}| + |P_{t,i}|)

  - Important constraints implemented:
      * For a given item i, days with actual A_{t,i} == 0 are excluded from the
        per-item time average (as described in the problem statement).
      * Only the first 7 unique dates from each TEST_XX.csv are used.
      * Items are matched by exact column name (e.g., "업장_메뉴명") between
        predictions and actuals.
      * Shop grouping is inferred from the item prefix before the first '_'.
      * Shops are averaged with user-provided weights (if any). If not provided,
        shops are averaged equally.

CLI usage examples:
  python smape_eval.py --pred "../0817_mljar_06003.csv" --test-dir "../data/test" --per-test
  python smape_eval.py --pred "../0817_mljar_06003.csv" --test-dir "../data/test" \
      --weights "담하=0.278,미라시아=0.278,화담숲주막=0.278,라그로타=0.166,느티나무 셀프BBQ=0,연회장=0,카페테리아=0,포레스트릿=0,화담숲카페=0"

Outputs overall score and (optionally) per-TEST scores.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple, Optional


def read_csv_utf8sig(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV handling BOM (utf-8-sig). Returns (headers, rows)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [((h or "").replace("\ufeff", "")) for h in (reader.fieldnames or [])]
        rows: List[Dict[str, str]] = []
        for r in reader:
            rows.append({((k or "").replace("\ufeff", "")): v for k, v in r.items()})
        return headers, rows


def smape(a: float, p: float) -> float:
    denom = abs(a) + abs(p)
    return 0.0 if denom == 0 else 2.0 * abs(p - a) / denom


def build_pred_mapping(pred_path: str) -> Tuple[List[str], Dict[str, Dict[int, Dict[str, float]]]]:
    """Parse the prediction CSV into a nested mapping:
    returns (item_columns, pred_by_test) where pred_by_test[test_id][day] -> {item: value}.
    Expects 영업일자 formatted like 'TEST_00+1일'.
    """
    headers, rows = read_csv_utf8sig(pred_path)
    if "영업일자" not in headers:
        raise ValueError("Prediction file must contain '영업일자' column.")
    item_cols = [h for h in headers if h != "영업일자"]

    pred_by_test: Dict[str, Dict[int, Dict[str, float]]] = {}
    pattern = re.compile(r"^TEST_(\d{2})\+(\d)일$")
    for row in rows:
        key = (row.get("영업일자") or "").strip()
        m = pattern.match(key)
        if not m:
            continue
        test_id, day_idx_str = m.group(1), m.group(2)
        day_idx = int(day_idx_str)
        items: Dict[str, float] = {}
        for col in item_cols:
            raw = (row.get(col, "") or "").strip()
            # Numbers may include commas; empty -> 0.0
            try:
                items[col] = float(raw.replace(",", "")) if raw != "" else 0.0
            except Exception:
                items[col] = 0.0
        pred_by_test.setdefault(test_id, {})[day_idx] = items

    return item_cols, pred_by_test


def compute_scores(
    pred_item_cols: List[str],
    pred_by_test: Dict[str, Dict[int, Dict[str, float]]],
    test_dir: str,
    shop_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute overall and per-TEST sMAPE scores.

    - Groups items by shop derived from the prefix before the first '_'.
    - For each TEST_XX.csv in test_dir, uses the first 7 unique dates.
    - For each item, averages over days with actual > 0.
    - For each shop, averages over items in the shop.
    - Overall is the mean of per-shop averages (equal shop weights) per test,
      then the mean of per-test scores.
    """
    # Infer shops from prediction item columns
    item_to_shop: Dict[str, str] = {
        col: (col.split("_", 1)[0] if "_" in col else col) for col in pred_item_cols
    }
    all_shops = sorted({item_to_shop[c] for c in pred_item_cols})

    per_test_score: Dict[str, float] = {}

    for test_path in sorted(glob.glob(os.path.join(test_dir, "TEST_*.csv"))):
        test_id = os.path.splitext(os.path.basename(test_path))[0].split("_")[1]

        # Read actuals and pivot to date -> {item: qty}
        headers, rows = read_csv_utf8sig(test_path)
        if not {"영업일자", "영업장명_메뉴명", "매출수량"}.issubset(set(headers)):
            continue

        dates = sorted({r["영업일자"] for r in rows})
        if len(dates) < 7:
            continue
        first7 = dates[:7]

        pivot: Dict[str, Dict[str, float]] = {d: {} for d in first7}
        for r in rows:
            d = r["영업일자"]
            if d not in pivot:
                continue
            item = (r["영업장명_메뉴명"] or "").strip()
            try:
                qty = float(r["매출수량"])  # numbers are small ints
            except Exception:
                qty = 0.0
            pivot[d][item] = pivot[d].get(item, 0.0) + qty

        actual_rows = [pivot[d] for d in first7]

        # Need 7-day predictions for this TEST id
        if test_id not in pred_by_test or any(k not in pred_by_test[test_id] for k in range(1, 8)):
            continue
        pred_rows7 = [pred_by_test[test_id][k] for k in range(1, 8)]

        # Items that appeared (at least once) in the 7-day window and exist in predictions
        actual_items = set().union(*[set(r.keys()) for r in actual_rows]) if actual_rows else set()
        common_items = [it for it in actual_items if it in item_to_shop]
        if not common_items:
            continue

        # Per-shop aggregation of per-item sMAPE averages (days with A==0 dropped)
        shop_to_item_avgs: Dict[str, List[float]] = {s: [] for s in all_shops}
        for it in common_items:
            shop_name = item_to_shop[it]
            day_vals: List[float] = []
            for day_idx in range(7):
                a = actual_rows[day_idx].get(it, 0.0)
                if a == 0:
                    continue
                p = pred_rows7[day_idx].get(it, 0.0)
                day_vals.append(smape(a, p))
            if day_vals:
                shop_to_item_avgs[shop_name].append(sum(day_vals) / len(day_vals))

        # Shop averages (skip shops without items)
        shop_avgs_map: Dict[str, float] = {s: (sum(v) / len(v)) for s, v in shop_to_item_avgs.items() if v}
        if not shop_avgs_map:
            continue
        # Weighted or equal average over shops present
        if shop_weights:
            total_w = sum(shop_weights.get(s, 0.0) for s in shop_avgs_map.keys())
            if total_w > 0:
                per_test_score[test_id] = sum((shop_weights.get(s, 0.0) / total_w) * v for s, v in shop_avgs_map.items())
            else:
                per_test_score[test_id] = sum(shop_avgs_map.values()) / len(shop_avgs_map)
        else:
            per_test_score[test_id] = sum(shop_avgs_map.values()) / len(shop_avgs_map)

    if not per_test_score:
        raise RuntimeError("No comparable data found. Check paths and file formats.")

    overall = sum(per_test_score.values()) / len(per_test_score)
    return overall, per_test_score


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute leaderboard-style sMAPE for predictions vs test CSVs.")
    ap.add_argument("--pred", required=True, help="Path to prediction CSV (columns: 영업일자 + item columns)")
    ap.add_argument("--test-dir", required=True, help="Directory containing TEST_*.csv files")
    ap.add_argument("--per-test", action="store_true", help="Print per-TEST scores as well as overall")
    ap.add_argument(
        "--weights",
        default=None,
        help=(
            "Optional shop weights mapping. Either a path to a JSON file containing {shop: weight}, "
            "or an inline string like '담하=0.278,미라시아=0.278,화담숲주막=0.278,라그로타=0.166,...'"
        ),
    )
    return ap.parse_args()


def _parse_weights_arg(weights_arg: Optional[str]) -> Optional[Dict[str, float]]:
    if not weights_arg:
        return None
    # If looks like a file path and exists, try JSON
    if os.path.exists(weights_arg):
        try:
            import json

            with open(weights_arg, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ensure float
            return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass
    # Fallback: parse inline "shop=weight,shop2=weight2"
    out: Dict[str, float] = {}
    for part in weights_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            continue
    return out if out else None


def main() -> None:
    args = parse_args()
    shop_weights = _parse_weights_arg(args.weights)
    item_cols, pred_map = build_pred_mapping(args.pred)
    overall, per_test = compute_scores(item_cols, pred_map, args.test_dir, shop_weights)

    if shop_weights:
        print(f"Overall (weighted shops, A>0 days only): {overall:.6f} ({overall*100:.3f}%)")
    else:
        print(f"Overall (equal shop weights, A>0 days only): {overall:.6f} ({overall*100:.3f}%)")
    if args.per_test:
        for k in sorted(per_test.keys()):
            v = per_test[k]
            print(f"TEST_{k}: {v:.6f} ({v*100:.3f}%)")


if __name__ == "__main__":
    main()


