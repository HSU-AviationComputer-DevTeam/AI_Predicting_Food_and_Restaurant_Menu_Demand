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
    print("/Users/sindong-u/coding/python/myenv/bin/pip install mljar-supervised")
    raise

BASE_DIR = "/Users/sindong-u/coding/python/Project/LgAImers/2기"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train", "train.csv")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_DIR = BASE_DIR

np.random.seed(42)

# -------------------------------
# Holiday calendar (provided)
# -------------------------------

HOLIDAY_DATES = pd.to_datetime([
    "2023-01-01","2023-01-21","2023-01-22","2023-01-23","2023-01-24",
    "2023-03-01","2023-05-05","2023-05-27","2023-05-29","2023-06-06",
    "2023-08-15","2023-09-28","2023-09-29","2023-09-30","2023-10-03",
    "2023-10-09","2023-12-25",
    "2024-01-01","2024-02-09","2024-02-10","2024-02-11","2024-02-12",
    "2024-03-01","2024-04-10","2024-05-05","2024-05-06","2024-05-15",
    "2024-06-06","2024-08-15","2024-09-16","2024-09-17","2024-09-18",
    "2024-10-03","2024-10-09","2024-12-25",
    "2025-01-01","2025-01-27","2025-01-28","2025-01-29","2025-01-30",
    "2025-03-01","2025-03-03","2025-05-05","2025-05-06","2025-06-03",
    "2025-06-06","2025-08-15","2025-10-03","2025-10-05","2025-10-06",
    "2025-10-07","2025-10-08","2025-10-09","2025-12-25",
])
HOLIDAY_SET = set(pd.DatetimeIndex(HOLIDAY_DATES).normalize())

# -------------------------------
# Shop weights (provided)
# -------------------------------

SHOP_WEIGHTS = {
    "담하": 0.181822,
    "미라시아": 0.363188,
    "화담숲주막": 0.001993,
    "라그로타": 0.180205,
    "느티나무 셀프BBQ": 0.085110,
    "연회장": 0.181703,
    "카페테리아": 0.001993,
    "포레스트릿": 0.001993,
    "화담숲카페": 0.001993,
}

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
    # additional cyclic for week-of-year
    df["weekofyear_sin"] = np.sin(2 * np.pi * df["weekofyear"].astype(float) / 52.0)
    df["weekofyear_cos"] = np.cos(2 * np.pi * df["weekofyear"].astype(float) / 52.0)
    # one-hot for day of week and month (robust seasonality)
    for d in range(7):
        df[f"dow_{d}"] = (df["dayofweek"] == d).astype(int)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    # holidays
    norm_dt = df["영업일자"].dt.normalize()
    df["is_holiday"] = norm_dt.isin(HOLIDAY_SET).astype(int)
    # neighbor-holiday effects (±1 day) and 2-day window
    df["is_holiday_m1"] = (norm_dt - pd.Timedelta(days=1)).isin(HOLIDAY_SET).astype(int)
    df["is_holiday_p1"] = (norm_dt + pd.Timedelta(days=1)).isin(HOLIDAY_SET).astype(int)
    df["is_holiday_window2"] = ((df["is_holiday"] == 1) | (df["is_holiday_m1"] == 1) | (df["is_holiday_p1"] == 1)).astype(int)
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
    # base shifted series
    s_shift = group[value_col].shift(1)
    # positive-only mask for robust rolling (zeros -> NaN)
    s_pos = s_shift.where(s_shift > 0, np.nan)
    for win in [7, 14, 28]:
        # standard mean (kept for backward-compat)
        df[f"ma_{win}"] = s_shift.rolling(win, min_periods=1).mean()
        # robust mean over positive values only
        df[f"ma_pos_{win}"] = s_pos.rolling(win, min_periods=1).mean()
    # rolling median (robust to spikes)
    df["median_7"] = s_pos.rolling(7, min_periods=1).median()
    # nonzero rate in recent window
    df["nonzero_rate_28"] = s_shift.gt(0).rolling(28, min_periods=1).mean()
    # exponentially weighted means (smoother)
    df["ewm_7"] = s_shift.ewm(span=7, adjust=False).mean()
    df["ewm_14"] = s_shift.ewm(span=14, adjust=False).mean()
    # 동일 요일 평균(최근 4주)
    dow_lags = [f"lag_{k}" for k in [7, 14, 21, 28]]
    df["dow_ma_4"] = df[dow_lags].mean(axis=1)
    # 단기 추세(최근 7일 평균 대비 그 이전 7일 평균)
    df["ma_7_prev"] = group[value_col].shift(8).rolling(7, min_periods=1).mean()
    df["trend_7"] = (df["ma_7"] / df["ma_7_prev"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["change_rate_7"] = (df[value_col] - df["lag_7"]) / (df["lag_7"].replace(0, np.nan))
    df["change_rate_7"] = df["change_rate_7"].replace([np.inf, -np.inf], 0).fillna(0)
    # last nonzero value and days since last sale
    df["last_pos_date"] = group["영업일자"].apply(lambda s: s.where(df.loc[s.index, value_col] > 0).ffill())
    df["days_since_last_sale"] = (df["영업일자"] - df["last_pos_date"]).dt.days.fillna(9999).astype(int)
    df.drop(columns=["last_pos_date"], inplace=True)
    df["last_nonzero_value"] = group[value_col].apply(lambda s: s.where(s > 0).ffill().shift(1))
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
        "year", "month", "day", "dayofweek", "is_weekend", "is_holiday", "quarter", "weekofyear", "dayofyear",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "weekofyear_sin", "weekofyear_cos",
        # one-hot dow/month
        "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6",
        "month_1","month_2","month_3","month_4","month_5","month_6","month_7","month_8","month_9","month_10","month_11","month_12",
        # holiday windows
        "is_holiday_m1","is_holiday_p1","is_holiday_window2",
        # lags & rolls
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "ma_7", "ma_14", "ma_28", "ma_pos_7", "ma_pos_14", "ma_pos_28", "median_7", "nonzero_rate_28", "ewm_7", "ewm_14", "dow_ma_4", "trend_7", "change_rate_7", "days_since_last_sale", "last_nonzero_value",
        # static
        "영업장명_메뉴명", "영업장명", "first_sale_month", "peak_month", "is_new_menu", "is_discontinued",
    ]

    # Drop rows without sufficient history
    train_feat = train_feat.dropna(subset=["lag_1", "lag_7"]).copy()

    X = train_feat[feature_cols]
    y = train_feat["매출수량"].astype(float)

    # Build sample weights EXACTLY as provided:
    #  - base downweight for zero-actual rows (LB excludes zeros)
    #  - multiply by provided shop weights (no rescaling, no extra boosts)
    venue = train_feat["영업장명"].astype(str)
    base_w = np.where(y <= 0, 0.0, 1.0)
    # Training-time relaxed weights: clamp to min 0.05 and normalize to mean=1.0
    _weights_for_train = {k: max(float(v), 0.05) for k, v in SHOP_WEIGHTS.items()}
    _mean_w = float(np.mean(list(_weights_for_train.values()))) if _weights_for_train else 1.0
    if _mean_w > 0:
        _weights_for_train = {k: (v / _mean_w) for k, v in _weights_for_train.items()}
    venue_w = venue.map(_weights_for_train).astype(float).fillna(1.0).values
    sample_weight = base_w * venue_w

    # Use log1p target for more stable training on counts
    y_train = np.log1p(y.clip(lower=0))

    # AutoML settings (use holdout split without shuffling)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"mljar_results_{run_tag}")
    os.makedirs(results_dir, exist_ok=True)
    base_automl_kwargs = dict(
        results_path=results_dir,
        mode="Compete",
        algorithms=["LightGBM", "CatBoost"],
        eval_metric="mae",
        total_time_limit=60 * 500,  # 500 minutes cap
        model_time_limit=60 * 15,
        validation_strategy={
            "validation_type": "split",   # keep simple and deterministic
            "train_ratio": 0.90,
            "shuffle": False,
        },
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        random_state=42,
    )

    # -------------------------------
    # Direct models: train 7 horizon-specific models
    # -------------------------------
    print("[INFO] Training AutoML models (Direct 1..7 days) ...")
    models_by_h = {}
    for h in range(1, 8):
        # Target at horizon h
        y_target = train_feat.groupby("영업장명_메뉴명")["매출수량"].shift(-h)
        dfh = train_feat.copy()
        dfh["y_target"] = y_target
        # drop rows with missing lag features or target
        dfh = dfh.dropna(subset=["lag_1", "lag_7", "y_target"]).copy()
        Xh = dfh[feature_cols]
        yh = dfh["y_target"].astype(float)
        # sample weights at target horizon (A=0 제외)
        venue_h = dfh["영업장명"].astype(str)
        base_w_h = np.where(yh <= 0, 0.0, 1.0)
        venue_w_h = venue_h.map(_weights_for_train).astype(float).fillna(1.0).values
        sample_weight_h = base_w_h * venue_w_h
        yh_log = np.log1p(yh.clip(lower=0))
        # separate results path per horizon
        kwargs = dict(base_automl_kwargs)
        kwargs["results_path"] = os.path.join(results_dir, f"h{h}")
        os.makedirs(kwargs["results_path"], exist_ok=True)
        automl_h = AutoML(**kwargs)
        automl_h.fit(Xh, yh_log, sample_weight=sample_weight_h)
        models_by_h[h] = automl_h

    # -------------------------------
    # Learn simple per-horizon blend weights on recent validation window
    # -------------------------------
    def build_blend_examples(h: int, lookback_days: int = 56):
        records = []  # tuples of (y_true, yhat_model, dow_avg, ma7)
        # take recent window from train_df per menu
        for menu, g in train_df.groupby("영업장명_메뉴명"):
            g = g.sort_values("영업일자").copy()
            if len(g) < 35:
                continue
            last_date = g["영업일자"].max()
            start_date = last_date - pd.Timedelta(days=lookback_days)
            sub = g[g["영업일자"] >= start_date]
            for d in sub["영업일자"].values:
                d = pd.to_datetime(d)
                target_day = d + pd.Timedelta(days=h)
                # target
                yt = g.loc[g["영업일자"] == target_day, "매출수량"]
                if len(yt) == 0:
                    continue
                y_true = float(yt.values[0])
                # history up to d
                hist = g[g["영업일자"] <= d][["영업일자", "영업장명_메뉴명", "매출수량"]].copy()
                feat = build_future_row(menu, target_day, hist, lifecycle_map)
                Xb = feat[feature_cols]
                yhat_log = float(models_by_h[h].predict(Xb)[0])
                yhat_model = float(np.expm1(yhat_log))
                # recent stats
                recent_tail = hist.sort_values("영업일자").tail(28)
                ma7 = float(recent_tail["매출수량"].tail(7).mean()) if len(recent_tail) > 0 else 0.0
                dow = target_day.dayofweek
                dow_mask = recent_tail["영업일자"].dt.dayofweek == dow
                dow_avg = float(recent_tail.loc[dow_mask, "매출수량"].mean()) if dow_mask.any() else ma7
                records.append((y_true, yhat_model, dow_avg, ma7))
        return records

    def grid_learn_weights(recs, grid_step: float = 0.05):
        # minimize sMAPE-like loss; constraints: w_model>=0, w_dow>=0, w_ma7 = 1-w_model-w_dow >=0
        best = (0.7, 0.2, 1.0)
        best_loss = float("inf")
        if not recs:
            return {h: (0.70, 0.20) for h in range(1, 8)}
        eps = 1e-6
        for wm in np.arange(0.5, 0.86, grid_step):
            for wd in np.arange(0.1, 0.36, grid_step):
                wma = 1.0 - wm - wd
                if wma < 0:
                    continue
                loss = 0.0
                n = 0
                for y, m, d, a7 in recs:
                    # exclude zeros to mimic LB
                    if y <= 0:
                        continue
                    yhat = wm * m + wd * d + wma * a7
                    denom = abs(y) + abs(yhat) + eps
                    loss += 2.0 * abs(y - yhat) / denom
                    n += 1
                if n == 0:
                    continue
                loss /= n
                if loss < best_loss:
                    best_loss = loss
                    best = (wm, wd, wma)
        return best[:2]

    blend_w = {}
    for h in range(1, 8):
        recs = build_blend_examples(h)
        wm, wd = grid_learn_weights(recs)
        blend_w[h] = (float(wm), float(wd))

    # Persist small config used for inference
    cfg = {
        "feature_cols": feature_cols,
        "use_log1p": True,
    }
    with open(os.path.join(OUTPUT_DIR, "mljar_inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return models_by_h, lifecycle_map, feature_cols, blend_w


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

    # ------------------------------
    # Robust rolling/rarity features for inference (match training names)
    # ------------------------------
    recent28 = history[history["영업일자"] < future_date].sort_values("영업일자").tail(28)
    values = recent28["매출수량"].astype(float).values if len(recent28) > 0 else np.array([])
    def masked_mean_last(vals: np.ndarray, win: int) -> float:
        if vals.size == 0:
            return 0.0
        seg = vals[-win:]
        seg_pos = seg[seg > 0]
        return float(seg_pos.mean()) if seg_pos.size > 0 else 0.0
    def median_last(vals: np.ndarray, win: int) -> float:
        if vals.size == 0:
            return 0.0
        seg = vals[-win:]
        seg_pos = seg[seg > 0]
        return float(np.median(seg_pos)) if seg_pos.size > 0 else 0.0
    # ma_pos_*
    tmp["ma_pos_7"] = masked_mean_last(values, 7)
    tmp["ma_pos_14"] = masked_mean_last(values, 14)
    tmp["ma_pos_28"] = masked_mean_last(values, 28)
    # median_7
    tmp["median_7"] = median_last(values, 7)
    # nonzero_rate_28
    if values.size == 0:
        tmp["nonzero_rate_28"] = 0.0
    else:
        tmp["nonzero_rate_28"] = float((values > 0).mean())
    # ewm_7 / ewm_14 (take last EWM value)
    if values.size == 0:
        tmp["ewm_7"] = 0.0
        tmp["ewm_14"] = 0.0
    else:
        s = pd.Series(values)
        tmp["ewm_7"] = float(s.ewm(span=7, adjust=False).mean().iloc[-1])
        tmp["ewm_14"] = float(s.ewm(span=14, adjust=False).mean().iloc[-1])
    # days_since_last_sale / last_nonzero_value
    hist_pos = history[(history["영업일자"] < future_date) & (history["매출수량"].astype(float) > 0)].sort_values("영업일자")
    if len(hist_pos) == 0:
        tmp["days_since_last_sale"] = int(9999)
        tmp["last_nonzero_value"] = float(0.0)
    else:
        last_row = hist_pos.iloc[-1]
        tmp["days_since_last_sale"] = int((future_date - pd.to_datetime(last_row["영업일자"]).normalize()).days)
        tmp["last_nonzero_value"] = float(last_row["매출수량"])

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


def forecast_7_days(automl_or_models, lifecycle_map: dict, feature_cols: list, test_csv_path: str, blend_w: dict | None = None) -> dict:
    """Return dict: {menu: [7 preds]} for one TEST file."""
    df = pd.read_csv(test_csv_path)
    df = parse_date(df)

    # Build initial history from provided test file (assumed last 28 days of actuals)
    history = df[["영업일자", "영업장명_메뉴명", "매출수량"]].copy()

    preds_per_menu = {}
    menus = sorted(df["영업장명_메뉴명"].unique())

    last_date = history["영업일자"].max()
    # Determine if we have direct models
    is_direct = isinstance(automl_or_models, dict)
    models_by_h = automl_or_models if is_direct else None

    # simple per-horizon blend weights (can be tuned/learned offline)
    if blend_w is None:
        blend_w = {h: (0.70, 0.20) for h in range(1, 8)}  # default

    for menu in menus:
        preds = []
        # Use the same immutable history for direct forecasting (no recursion)
        menu_hist_full = history[history["영업장명_메뉴명"] == menu].copy()
        for step in range(1, 8):
            future_date = last_date + timedelta(days=step)
            feat_row = build_future_row(menu, future_date, menu_hist_full, lifecycle_map)
            Xf = feat_row[feature_cols].copy()
            if is_direct:
                yhat_log = float(models_by_h[step].predict(Xf)[0])
            else:
                yhat_log = float(automl_or_models.predict(Xf)[0])
            # inverse transform from log1p
            yhat_model = float(np.expm1(yhat_log))
            # 최근 히스토리 기반 보정값들
            recent = menu_hist_full.sort_values("영업일자")
            recent_tail = recent[recent["영업일자"] < future_date].tail(28)
            ma7 = float(recent_tail["매출수량"].tail(7).mean()) if len(recent_tail) > 0 else 0.0
            # 요일 평균(최근 4주)
            dow = future_date.dayofweek
            dow_mask = recent_tail["영업일자"].dt.dayofweek == dow
            dow_avg = float(recent_tail.loc[dow_mask, "매출수량"].mean()) if dow_mask.any() else ma7
            shop = str(menu).split("_")[0]
            # Learned/static meta-blend per horizon (horizon-based floors)
            w_model, w_dow = blend_w.get(step, (0.70, 0.20))
            # enforce horizon-wise minimums (reduce over-conservatism)
            if step <= 3:
                w_model = max(w_model, 0.65)
            elif step <= 5:
                w_model = max(w_model, 0.60)
            else:
                w_model = max(w_model, 0.55)
            w_ma7 = max(0.0, 1.0 - w_model - w_dow)
            yhat = w_model * yhat_model + w_dow * dow_avg + w_ma7 * ma7
            
            # 하한선(floor): 음료는 최근 14일에 비영 1회 이상 또는 dow_avg>=1 이면 최소 1 보장
            bev_kw = ["콜라", "스프라이트", "아메리카노", "카페라떼", "생수", "맥주", "커피", "라떼"]
            has_bev = any(k in menu for k in bev_kw)
            dense = float(feat_row["nonzero_rate_28"].values[0]) if "nonzero_rate_28" in feat_row.columns else 0.0
            # recent14 positive exist
            recent14 = recent[recent["영업일자"] < future_date].tail(14)
            pos14 = int((recent14["매출수량"].astype(float) > 0).any()) if len(recent14) > 0 else 0
            floor_base = 0.0 if ma7 <= 0 else 0.15 * ma7
            if has_bev and (pos14 == 1 or dow_avg >= 1.0):
                floor_base = max(floor_base, 1.0)
            yhat = max(yhat, floor_base)
            
            # 상한선 캡(급등 방지): 기본 p95, CV>1.5일 때만 p90, 캡 계수 1.15
            if len(recent_tail) > 0:
                hist_vals = recent_tail["매출수량"].astype(float).values
                mu = float(np.mean(hist_vals)) if hist_vals.size > 0 else 0.0
                sigma = float(np.std(hist_vals)) if hist_vals.size > 0 else 0.0
                cv = (sigma / mu) if mu > 0 else 0.0
                perc = 90 if (cv > 1.5) else 95
                pxx = float(np.percentile(hist_vals, perc))
                hist_max = float(np.max(hist_vals))
            else:
                pxx = yhat
                hist_max = yhat
            cap = max(1.0, min(pxx * 1.15, hist_max * 1.20, (ma7 * 3.0 if ma7 > 0 else hist_max * 1.20)))

            # 희소 품목 수축: 대여료/단체식 등은 과대 방지 (안전 계수)
            sparse_kw = ["대여료", "단체식"]
            if any(k in menu for k in sparse_kw) and dense < 0.25:
                yhat = min(yhat, max(dow_avg, 0.0) * 0.90 if dow_avg > 0 else yhat * 0.90)

            # 카페테리아 고위험 메뉴 보수 수축: 희소(dense<0.2) 또는 dow_avg==0일 때만
            risky_cafe_menu = ["짜장면", "짬뽕", "짬뽕밥", "치즈돈까스", "수제 등심 돈까스", "공깃밥(추가)"]
            if shop == "카페테리아" and any(k in menu for k in risky_cafe_menu) and step >= 3:
                if (dense < 0.2) or (dow_avg <= 0):
                    yhat *= 0.93

            # 브런치: 평일에는 보수적으로 (주말/공휴일은 유지)
            if "브런치" in menu and feat_row["is_weekend"].values[0] == 0:
                yhat *= 0.92

            # 연회장 시설: 회의실/컨벤션은 대부분 0/1, OPUS/Grand Ballroom은 최대 2
            if shop == "연회장" and any(k in menu for k in ["Conference", "Convention Hall", "Grand Ballroom", "OPUS"]):
                hard_cap = 2 if ("Grand Ballroom" in menu or "OPUS" in menu) else 1
                yhat = min(yhat, hard_cap)

            yhat = min(yhat, cap)
            # Business constraints: clip >= 0 and round to int
            yhat = int(max(0, round(yhat)))
            preds.append(yhat)
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
    models_by_h, lifecycle_map, feature_cols, blend_w = train_model()

    print("[STEP] Forecasting 7 days for each TEST file ...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    for tf in test_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]  # e.g., TEST_00
        print(f"  - Predicting for {test_id} ...")
        menu_preds = forecast_7_days(models_by_h, lifecycle_map, feature_cols, tf, blend_w)
        all_test_preds[test_id] = menu_preds

    print("[STEP] Building submission ...")
    sub = build_submission(all_test_preds)
    out_path = os.path.join(OUTPUT_DIR, f"submission_mljar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
