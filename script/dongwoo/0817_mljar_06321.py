import os
import sys
import glob
import json
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import mljar-supervised; if missing, print instruction
try:
    from supervised.automl import AutoML
except Exception as e:
    print("[INFO] mljar-supervised not found. Please install it:")
    print("       /Users/sindong-u/coding/python/myenv/bin/pip install mljar-supervised")
    raise

BASE_DIR = "/Users/sindong-u/coding/python/Project/LgAImers/2기"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train", "train.csv")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_DIR = BASE_DIR

np.random.seed(42)

# -------------------------------
# Utilities
# -------------------------------

def parse_date(df: pd.DataFrame, col: str = "영업일자") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["영업일자"].dt.year
    df["month"] = df["영업일자"].dt.month
    df["day"] = df["영업일자"].dt.day
    df["dayofweek"] = df["영업일자"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["quarter"] = df["영업일자"].dt.quarter
    df["weekofyear"] = df["영업일자"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["영업일자"].dt.dayofyear
    # cyclic encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_lifecycle_maps(train_df: pd.DataFrame):
    """Compute per-menu lifecycle statistics from training data."""
    life = {}
    grouped = train_df[train_df["매출수량"] > 0].groupby("영업장명_메뉴명")
    first_sale = grouped["영업일자"].min()
    last_sale = grouped["영업일자"].max()
    avg_sales = grouped["매출수량"].mean()
    peak_month = grouped.apply(lambda g: g.groupby(g["영업일자"].dt.month)["매출수량"].sum().idxmax())

    for menu in train_df["영업장명_메뉴명"].unique():
        fs = first_sale.get(menu, pd.NaT)
        ls = last_sale.get(menu, pd.NaT)
        am = float(avg_sales.get(menu, 0.0))
        pm = int(peak_month.get(menu, 6)) if not pd.isna(peak_month.get(menu, np.nan)) else 6
        pattern = "regular"
        if pd.notna(fs) and fs >= pd.Timestamp("2023-06-01"):
            pattern = "new_menu"
        if pd.notna(ls) and ls <= pd.Timestamp("2024-01-31"):
            pattern = "possibly_discontinued"
        life[menu] = {
            "first_sale": fs,
            "last_sale": ls,
            "avg_sales": am,
            "peak_month": pm,
            "pattern": pattern,
        }
    return life


def add_group_lag_features(df: pd.DataFrame, value_col: str = "매출수량") -> pd.DataFrame:
    df = df.sort_values(["영업장명_메뉴명", "영업일자"]).copy()
    group = df.groupby("영업장명_메뉴명", group_keys=False)
    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = group[value_col].shift(lag)
    for win in [7, 14, 28]:
        df[f"ma_{win}"] = group[value_col].shift(1).rolling(win, min_periods=1).mean()
    # 동일 요일 평균(최근 4주)
    dow_lags = [f"lag_{k}" for k in [7, 14, 21, 28]]
    df["dow_ma_4"] = df[dow_lags].mean(axis=1)
    # 단기 추세(최근 7일 평균 대비 그 이전 7일 평균)
    df["ma_7_prev"] = group[value_col].shift(8).rolling(7, min_periods=1).mean()
    df["trend_7"] = (df["ma_7"] / df["ma_7_prev"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["change_rate_7"] = (df[value_col] - df["lag_7"]) / (df["lag_7"].replace(0, np.nan))
    df["change_rate_7"] = df["change_rate_7"].replace([np.inf, -np.inf], 0).fillna(0)
    return df


def add_static_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["영업장명"] = df["영업장명_메뉴명"].str.split("_").str[0]
    # lifecycle encodings
    fs_month = []
    peak_month = []
    new_flag = []
    disc_flag = []
    for m in df["영업장명_메뉴명"].values:
        info = lifecycle_map.get(m, None)
        if info is None:
            fs_month.append(0)
            peak_month.append(6)
            new_flag.append(0)
            disc_flag.append(0)
        else:
            fs_month.append(0 if pd.isna(info["first_sale"]) else int(info["first_sale"].month))
            peak_month.append(int(info["peak_month"]))
            new_flag.append(1 if info["pattern"] == "new_menu" else 0)
            disc_flag.append(1 if info["pattern"] == "possibly_discontinued" else 0)
    df["first_sale_month"] = fs_month
    df["peak_month"] = peak_month
    df["is_new_menu"] = new_flag
    df["is_discontinued"] = disc_flag
    return df


def make_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    df = parse_date(df)
    df = add_date_features(df)
    df = add_group_lag_features(df)
    df = add_static_features(df, lifecycle_map)
    # Clean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return df


# -------------------------------
# Load train and build model
# -------------------------------

def train_model() -> tuple:
    print("[INFO] Loading train data ...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = parse_date(train_df)

    lifecycle_map = build_lifecycle_maps(train_df)

    # Build features
    train_feat = make_features(train_df, lifecycle_map)

    feature_cols = [
        # time
        "year", "month", "day", "dayofweek", "is_weekend", "quarter", "weekofyear", "dayofyear",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        # lags & rolls
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "ma_7", "ma_14", "ma_28", "dow_ma_4", "trend_7", "change_rate_7",
        # static
        "영업장명_메뉴명", "영업장명", "first_sale_month", "peak_month", "is_new_menu", "is_discontinued",
    ]

    # Drop rows without sufficient history
    train_feat = train_feat.dropna(subset=["lag_1", "lag_7"]).copy()

    X = train_feat[feature_cols]
    y = train_feat["매출수량"].astype(float)

    # Build sample weights: downweight zero actuals (ignored in LB), upweight key venues
    venue = train_feat["영업장명"].astype(str)
    w = np.where(y <= 0, 0.1, 1.0)  # zero actuals contribute less
    vboost = np.where(venue.str.contains("담하|미라시아"), 2.0, 1.0)
    sample_weight = w * vboost

    # Use log1p target for more stable training on counts
    y_train = np.log1p(y.clip(lower=0))

    # AutoML settings (use holdout split without shuffling)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"mljar_results_{run_tag}")
    automl = AutoML(
        results_path=results_dir,
        mode="Compete",
        algorithms=["LightGBM", "CatBoost"],
        eval_metric="mae",
        total_time_limit=60 * 20,  # 20 minutes cap
        model_time_limit=60 * 3,
        validation_strategy={
            "validation_type": "split",   # holdout split
            "train_ratio": 0.85,
            "shuffle": False,
        },
        random_state=42,
    )

    print("[INFO] Training AutoML model ...")
    automl.fit(X, y_train, sample_weight=sample_weight)

    # Persist small config used for inference
    cfg = {
        "feature_cols": feature_cols,
        "use_log1p": True,
    }
    with open(os.path.join(OUTPUT_DIR, "mljar_inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return automl, lifecycle_map, feature_cols


# -------------------------------
# Forecasting utilities
# -------------------------------

def build_future_row(menu: str, future_date: pd.Timestamp, history: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    """Construct one-row dataframe of features for a specific menu & date using history (which contains 매출수량).
    history must contain rows for this menu with columns: 영업일자, 매출수량.
    """
    hist = history[history["영업장명_메뉴명"] == menu].sort_values("영업일자").copy()
    row = {
        "영업일자": future_date,
        "영업장명_메뉴명": menu,
    }
    tmp = pd.DataFrame([row])
    tmp = parse_date(tmp)
    tmp = add_date_features(tmp)

    # lags from history
    def get_lag(days: int):
        target_day = future_date - timedelta(days=days)
        v = hist.loc[hist["영업일자"] == target_day, "매출수량"]
        return float(v.values[0]) if len(v) > 0 else np.nan

    for lag in [1, 7, 14, 21, 28]:
        tmp[f"lag_{lag}"] = get_lag(lag)

    # rolling means
    for win in [7, 14, 28]:
        past_start = future_date - timedelta(days=win)
        mask = (hist["영업일자"] < future_date) & (hist["영업일자"] >= past_start)
        vals = hist.loc[mask, "매출수량"].astype(float).values
        tmp[f"ma_{win}"] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    # same-day-of-week average over last 4 weeks
    l7 = tmp["lag_7"].values[0]
    l14 = tmp["lag_14"].values[0]
    l21 = tmp["lag_21"].values[0]
    l28 = tmp["lag_28"].values[0]
    lag_vals = [v for v in [l7, l14, l21, l28] if not pd.isna(v)]
    tmp["dow_ma_4"] = float(np.mean(lag_vals)) if len(lag_vals) > 0 else 0.0

    # short-term trend: current 7d mean vs previous 7d mean
    # current 7d mean is ma_7 already computed excluding future_date
    ma7_now = tmp["ma_7"].values[0]
    prev_start = future_date - timedelta(days=14)
    prev_end = future_date - timedelta(days=8)
    prev_mask = (hist["영업일자"] <= prev_end) & (hist["영업일자"] >= prev_start)
    prev_vals = hist.loc[prev_mask, "매출수량"].astype(float).values
    ma7_prev = float(np.mean(prev_vals)) if len(prev_vals) > 0 else np.nan
    if pd.isna(ma7_prev) or ma7_prev == 0:
        tmp["trend_7"] = 1.0
    else:
        tmp["trend_7"] = float(ma7_now / ma7_prev)

    # change rate vs 7 days ago
    lag7 = tmp["lag_7"].values[0]
    last = tmp["lag_1"].values[0]
    if pd.isna(lag7) or lag7 == 0:
        tmp["change_rate_7"] = 0.0
    else:
        tmp["change_rate_7"] = float((last - lag7) / lag7)

    # static features
    tmp["영업장명"] = menu.split("_")[0]
    info = lifecycle_map.get(menu, None)
    if info is None:
        tmp["first_sale_month"] = 0
        tmp["peak_month"] = 6
        tmp["is_new_menu"] = 0
        tmp["is_discontinued"] = 0
    else:
        tmp["first_sale_month"] = 0 if pd.isna(info["first_sale"]) else int(info["first_sale"].month)
        tmp["peak_month"] = int(info["peak_month"]) if not pd.isna(info["peak_month"]) else 6
        tmp["is_new_menu"] = 1 if info["pattern"] == "new_menu" else 0
        tmp["is_discontinued"] = 1 if info["pattern"] == "possibly_discontinued" else 0

    # Clean
    num_cols = tmp.select_dtypes(include=[np.number]).columns
    tmp[num_cols] = tmp[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return tmp


def forecast_7_days(automl: AutoML, lifecycle_map: dict, feature_cols: list, test_csv_path: str) -> dict:
    """Return dict: {menu: [7 preds]} for one TEST file."""
    df = pd.read_csv(test_csv_path)
    df = parse_date(df)

    # Build initial history from provided test file (assumed last 28 days of actuals)
    history = df[["영업일자", "영업장명_메뉴명", "매출수량"]].copy()

    preds_per_menu = {}
    menus = sorted(df["영업장명_메뉴명"].unique())

    last_date = history["영업일자"].max()
    for menu in menus:
        preds = []
        # We copy history per menu to update iteratively
        menu_hist = history[history["영업장명_메뉴명"] == menu].copy()
        for step in range(1, 8):
            future_date = last_date + timedelta(days=step)
            feat_row = build_future_row(menu, future_date, menu_hist, lifecycle_map)
            Xf = feat_row[feature_cols].copy()
            yhat_log = float(automl.predict(Xf)[0])
            # inverse transform from log1p
            yhat_model = float(np.expm1(yhat_log))
            # 최근 히스토리 기반 보정값들
            recent = menu_hist.sort_values("영업일자")
            recent_tail = recent[recent["영업일자"] < future_date].tail(28)
            ma7 = float(recent_tail["매출수량"].tail(7).mean()) if len(recent_tail) > 0 else 0.0
            # 요일 평균(최근 4주) - 평균 대신 중앙값으로 이상치 영향 완화
            dow = future_date.dayofweek
            dow_mask = recent_tail["영업일자"].dt.dayofweek == dow
            same_dow_vals = recent_tail.loc[dow_mask, "매출수량"].tail(4).values
            if len(same_dow_vals) > 0:
                dow_avg = float(np.median(same_dow_vals))
            else:
                dow_avg = ma7

            # 변동성에 따른 가중치 (주말 가중 조정)
            m = float(recent_tail["매출수량"].mean()) if len(recent_tail) > 0 else 0.0
            s = float(recent_tail["매출수량"].std()) if len(recent_tail) > 1 else 0.0
            cv = (s / m) if m > 0 else 1.0
            is_weekend = (dow >= 5)
            if cv > 1.0:
                w_model = 0.55 if is_weekend else 0.60
                w_dow = 0.30 if is_weekend else 0.25
            else:
                w_model = 0.70 if is_weekend else 0.75
                w_dow = 0.20 if is_weekend else 0.15
            w_ma7 = max(0.0, 1.0 - w_model - w_dow)
            yhat = w_model * yhat_model + w_dow * dow_avg + w_ma7 * ma7

            # 라이프사이클 보정: 피크월 보너스, 단종 의심시 감쇠
            info = lifecycle_map.get(menu, None)
            if info is not None:
                try:
                    peak_m = int(info.get("peak_month", 6))
                except Exception:
                    peak_m = 6
                if future_date.month == peak_m:
                    yhat *= 1.05  # 피크월 소폭 상향
                last_sale = info.get("last_sale", pd.NaT)
                if info.get("pattern", "regular") == "possibly_discontinued" and pd.notna(last_sale):
                    # 마지막 판매일 이후 경과에 따라 감쇠 (60일 이후 급감)
                    days_since = (future_date - pd.to_datetime(last_sale)).days
                    if days_since > 60:
                        yhat *= 0.6
                    elif days_since > 30:
                        yhat *= 0.8

            # 하한선(floor) 적용: 저변 수요 품목의 과도한 0 예측 방지
            keywords = ["콜라", "스프라이트", "아메리카노", "카페라떼", "생수", "맥주", "커피", "라떼"]
            has_floor = any(k in menu for k in keywords)
            floor_base = 0.0 if ma7 <= 0 else 0.15 * ma7
            if has_floor:
                floor_base = max(floor_base, 1.0)
            yhat = max(yhat, floor_base)

            # 제로-스트릭 백오프: 최근 연속 0이 길면 보수적 예측
            zero_streak = 0
            for val in reversed(recent_tail["매출수량"].tolist()):
                if val == 0:
                    zero_streak += 1
                else:
                    break
            if zero_streak >= 5 and ma7 < 0.5:
                yhat = min(yhat, 1.0)

            # 상한선 캡(급등 방지)
            hist_max = float(recent_tail["매출수량"].max()) if len(recent_tail) > 0 else yhat
            cap = max(1.0, min(hist_max * 1.2, (ma7 * 3.0 if ma7 > 0 else hist_max * 1.2)))
            yhat = min(yhat, cap)

            # Business constraints: clip >= 0 and round to int
            yhat = int(max(0, round(yhat)))
            preds.append(yhat)
            # Append to history for next step
            menu_hist = pd.concat([
                menu_hist,
                pd.DataFrame({
                    "영업일자": [future_date],
                    "영업장명_메뉴명": [menu],
                    "매출수량": [yhat],
                })
            ], ignore_index=True)
        preds_per_menu[menu] = preds
    return preds_per_menu


# -------------------------------
# Submission builder
# -------------------------------

def build_submission(all_test_preds: dict) -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sub = sample.copy()
    # Fill using mapping from test file name to rows in sample
    # sample index like: TEST_00+1일 ... TEST_09+7일
    for test_id, menu_to_preds in all_test_preds.items():
        for day in range(1, 8):
            row_label = f"{test_id}+{day}일"
            if row_label in sub["영업일자"].values:
                ridx = sub.index[sub["영업일자"] == row_label][0]
                for menu, preds in menu_to_preds.items():
                    if menu in sub.columns:
                        sub.at[ridx, menu] = int(preds[day - 1])
    
    # Optional meta-blending with previous submissions to reduce variance
    try:
        prev_a_path = os.path.join(OUTPUT_DIR, "0817_mljar_0603.csv")
        prev_b_path = os.path.join(OUTPUT_DIR, "0817_mljar_064.csv")
        blend_sources = []
        if os.path.exists(prev_a_path):
            blend_sources.append(pd.read_csv(prev_a_path))
        if os.path.exists(prev_b_path):
            blend_sources.append(pd.read_csv(prev_b_path))
        if blend_sources:
            # Align columns
            cols = [c for c in sub.columns if c in set.intersection(*[set(df.columns) for df in [sub] + blend_sources])]
            tmp = sub[cols].copy().astype({c: int for c in cols if c != "영업일자"})
            frames = [df[cols].copy() for df in blend_sources]
            # Weighted blend: current 0.6, best(0603) 0.3, other 0.1 (if present)
            weights = []
            for df in frames:
                if os.path.samefile(prev_a_path, os.path.join(OUTPUT_DIR, os.path.basename(df.columns.name or prev_a_path))) if False else False:
                    weights.append(0.3)
            # Fallback: assign weights by order
            if not weights:
                weights = [0.3] + ([0.1] * (len(frames) - 1))
            # Compute
            blend_numeric = tmp.copy()
            for idx, df in enumerate(frames):
                for c in cols:
                    if c == "영업일자":
                        continue
                    blend_numeric[c] = blend_numeric[c].astype(float) * (0.6 if idx == -1 else 1.0)  # placeholder
            # Simpler: direct formula
            final = sub.copy()
            if len(frames) == 1:
                final.update(((0.7 * sub[cols].select_dtypes(include=[np.number]) + 0.3 * frames[0][cols].select_dtypes(include=[np.number]))
                              .round().clip(lower=0)).astype(int))
            else:
                final.update(((0.6 * sub[cols].select_dtypes(include=[np.number]) + 0.3 * frames[0][cols].select_dtypes(include=[np.number]) + 0.1 * frames[1][cols].select_dtypes(include=[np.number]))
                              .round().clip(lower=0)).astype(int))
            sub = final
    except Exception:
        # If anything goes wrong, keep original sub
        pass
    
    # Ensure integers and non-negative
    for c in sub.columns:
        if c == "영업일자":
            continue
        sub[c] = sub[c].fillna(0).astype(float).clip(lower=0).round().astype(int)
    return sub


def main():
    start = time.time()
    print("[STEP] Training with mljar-supervised ...")
    automl, lifecycle_map, feature_cols = train_model()

    print("[STEP] Forecasting 7 days for each TEST file ...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    for tf in test_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]  # e.g., TEST_00
        print(f"  - Predicting for {test_id} ...")
        menu_preds = forecast_7_days(automl, lifecycle_map, feature_cols, tf)
        all_test_preds[test_id] = menu_preds

    print("[STEP] Building submission ...")
    sub = build_submission(all_test_preds)
    out_path = os.path.join(OUTPUT_DIR, f"submission_mljar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
