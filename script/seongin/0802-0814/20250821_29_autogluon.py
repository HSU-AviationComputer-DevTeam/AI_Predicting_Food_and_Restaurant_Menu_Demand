import os
import sys
import glob
import json
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

warnings.filterwarnings("ignore")

# 상단 공용 경로 설정
from pathlib import Path

BASE_ROOT = Path(r"C:\GitHubRepo\AI_Forecasting_Food_and_Restaurant_Menu_Demand")
BASE_DATA = BASE_ROOT / "data"
TRAIN_DIR = BASE_DATA / "train"
TEST_DIR  = BASE_DATA / "test"
SAMPLE_SUB_PATH = BASE_DATA / "sample_submission.csv"
OUTPUT_DIR = BASE_ROOT / "autogluon_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RULE_CSV = TRAIN_DIR / "assoc_rules_by_venue.csv"

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

# 업장 정비 기간 (모든 매출 0)
MAINTENANCE_DATES = pd.to_datetime([
    "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12",
    "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", 
    "2025-03-07", "2025-03-08", "2025-03-09", "2025-03-10"
])
MAINTENANCE_SET = set(pd.DatetimeIndex(MAINTENANCE_DATES).normalize())

# -------------------------------
# Shop weights (provided)
# -------------------------------

SHOP_WEIGHTS = {
    "담하": 0.278000,
    "미라시아": 0.278000,
    "화담숲주막": 0.218000,
    "라그로타": 0.214000,
    "느티나무 셀프BBQ": 0.000300,
    "연회장": 0.000300,
    "카페테리아": 0.000300,
    "포레스트릿": 0.000300,
    "화담숲카페": 0.010800,
}

# -------------------------------
# Utilities (기존과 동일)
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
    df["weekofyear_sin"] = np.sin(2 * np.pi * df["weekofyear"].astype(float) / 52.0)
    df["weekofyear_cos"] = np.cos(2 * np.pi * df["weekofyear"].astype(float) / 52.0)
    
    # one-hot for day of week and month
    for d in range(7):
        df[f"dow_{d}"] = (df["dayofweek"] == d).astype(int)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    
    # holidays
    norm_dt = df["영업일자"].dt.normalize()
    df["is_holiday"] = norm_dt.isin(HOLIDAY_SET).astype(int)
    df["is_holiday_m1"] = (norm_dt - pd.Timedelta(days=1)).isin(HOLIDAY_SET).astype(int)
    df["is_holiday_p1"] = (norm_dt + pd.Timedelta(days=1)).isin(HOLIDAY_SET).astype(int)
    df["is_holiday_window2"] = ((df["is_holiday"] == 1) | (df["is_holiday_m1"] == 1) | (df["is_holiday_p1"] == 1)).astype(int)
    
    # 정비 기간 플래그 추가
    df["is_maintenance"] = norm_dt.isin(MAINTENANCE_SET).astype(int)
    df["is_maintenance_m1"] = (norm_dt - pd.Timedelta(days=1)).isin(MAINTENANCE_SET).astype(int)
    df["is_maintenance_p1"] = (norm_dt + pd.Timedelta(days=1)).isin(MAINTENANCE_SET).astype(int)
    
    return df

def add_group_lag_features(df: pd.DataFrame, value_col: str = "매출수량") -> pd.DataFrame:
    df = df.sort_values(["영업장명_메뉴명", "영업일자"]).copy()
    group = df.groupby("영업장명_메뉴명", group_keys=False)
    
    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = group[value_col].shift(lag)
    
    s_shift = group[value_col].shift(1)
    s_pos = s_shift.where(s_shift > 0, np.nan)
    
    for win in [7, 14, 28]:
        df[f"ma_{win}"] = s_shift.rolling(win, min_periods=1).mean()
        df[f"ma_pos_{win}"] = s_pos.rolling(win, min_periods=1).mean()
    
    df["median_7"] = s_pos.rolling(7, min_periods=1).median()
    df["nonzero_rate_28"] = s_shift.gt(0).rolling(28, min_periods=1).mean()
    df["ewm_7"] = s_shift.ewm(span=7, adjust=False).mean()
    df["ewm_14"] = s_shift.ewm(span=14, adjust=False).mean()
    
    # 동일 요일 평균(최근 4주)
    dow_lags = [f"lag_{k}" for k in [7, 14, 21, 28]]
    df["dow_ma_4"] = df[dow_lags].mean(axis=1)
    
    # 단기 추세
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

def add_static_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["영업장명"] = df["영업장명_메뉴명"].str.split("_").str[0]
    
    # 생명주기 특성 추가
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
    
    # 메뉴 카테고리 분류
    df["menu_category"] = "other"
    df.loc[df["영업장명_메뉴명"].str.contains("콜라|스프라이트|생수|맥주|커피|라떼|아메리카노"), "menu_category"] = "beverage"
    df.loc[df["영업장명_메뉴명"].str.contains("BBQ|불고기|삼겹살|돈까스|짜장면|짬뽕"), "menu_category"] = "main_dish"
    df.loc[df["영업장명_메뉴명"].str.contains("공깃밥|밥|면"), "menu_category"] = "side_dish"
    df.loc[df["영업장명_메뉴명"].str.contains("대여료|세트|패키지"), "menu_category"] = "service"
    
    return df

def make_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    df = parse_date(df)
    df = add_date_features(df)
    df = add_group_lag_features(df)
    df = add_static_features(df, lifecycle_map)
    
    # 정비 기간 관련 특성 추가
    df["days_since_maintenance"] = 0
    df["days_until_maintenance"] = 0
    
    for menu in df["영업장명_메뉴명"].unique():
        menu_data = df[df["영업장명_메뉴명"] == menu].sort_values("영업일자")
        maintenance_dates = menu_data[menu_data["is_maintenance"] == 1]["영업일자"]
        
        for idx, row in menu_data.iterrows():
            current_date = row["영업일자"]
            
            # 가장 최근 정비일로부터의 일수
            past_maintenance = maintenance_dates[maintenance_dates < current_date]
            if len(past_maintenance) > 0:
                df.loc[idx, "days_since_maintenance"] = (current_date - past_maintenance.max()).days
            
            # 다음 정비일까지의 일수
            future_maintenance = maintenance_dates[maintenance_dates > current_date]
            if len(future_maintenance) > 0:
                df.loc[idx, "days_until_maintenance"] = (future_maintenance.min() - current_date).days
    
    # Clean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return df

# -------------------------------
# AutoGluon Model Training
# -------------------------------

def train_autogluon_model() -> TabularPredictor:
    print("[INFO] Loading train data ...")
    train_df = pd.read_csv(TRAIN_DIR / "train.csv")
    train_df = parse_date(train_df)

    # 생명주기 맵 생성
    lifecycle_map = build_lifecycle_maps(train_df)
    
    # Build features
    train_feat = make_features(train_df, lifecycle_map)
    
    # Drop rows without sufficient history
    train_feat = train_feat.dropna(subset=["lag_1", "lag_7"]).copy()
    
    # 완전한 특성 리스트
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
        "영업장명_메뉴명", "영업장명", "first_sale_month", "peak_month", "is_new_menu", "is_discontinued", "menu_category",
        # 정비 기간 관련
        "is_maintenance", "is_maintenance_m1", "is_maintenance_p1",
        "days_since_maintenance", "days_until_maintenance",
    ]
    
    X = train_feat[feature_cols]
    y = train_feat["매출수량"].astype(float)
    
    # Build sample weights
    venue = train_feat["영업장명"].astype(str)
    base_w = np.where(y <= 0, 0.0, 1.0)
    venue_w = venue.map(SHOP_WEIGHTS).astype(float).fillna(1.0).values
    sample_weight = base_w * venue_w
    
    # Use log1p target for more stable training
    y_train = np.log1p(y.clip(lower=0))
    
    # 가중치를 적용한 데이터 준비 (중복 샘플링 방식)
    weighted_indices = []
    for i, weight in enumerate(sample_weight):
        # 가중치에 따라 샘플 복제 (0이 아닌 가중치만)
        if weight > 0:
            repeat_count = max(1, int(weight * 10))  # 가중치를 10배 스케일링
            weighted_indices.extend([i] * repeat_count)
    
    # 가중치가 적용된 데이터셋 생성
    X_weighted = X.iloc[weighted_indices].reset_index(drop=True)
    y_weighted = y_train.iloc[weighted_indices].reset_index(drop=True)
    
    print(f"[INFO] Original data size: {len(X)}")
    print(f"[INFO] Weighted data size: {len(X_weighted)}")
    print(f"[INFO] Weight scaling factor: 10x")
    
    # Prepare data for AutoGluon
    train_data = X_weighted.copy()
    train_data['매출수량'] = y_weighted
    
    # AutoGluon hyperparameters for better performance
    hyperparameters = {
        'GBM': [
            {'num_boost_round': 1000, 'learning_rate': 0.1, 'max_depth': 8, 'num_leaves': 31},
            {'num_boost_round': 1500, 'learning_rate': 0.05, 'max_depth': 10, 'num_leaves': 63}
        ],
        'CAT': [
            {'iterations': 1000, 'learning_rate': 0.1, 'depth': 8, 'l2_leaf_reg': 3},
            {'iterations': 1500, 'learning_rate': 0.05, 'depth': 10, 'l2_leaf_reg': 5}
        ],
        'XGB': [
            {'n_estimators': 1000, 'learning_rate': 0.1, 'max_depth': 8, 'subsample': 0.8},
            {'n_estimators': 1500, 'learning_rate': 0.05, 'max_depth': 10, 'subsample': 0.9}
        ],
        'RF': [
            {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
            {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3}
        ],
        'XT': [
            {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
            {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3}
        ]
    }
    
    print("[INFO] Training AutoGluon model ...")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = OUTPUT_DIR / f"autogluon_{run_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    predictor = TabularPredictor(
        label='매출수량',
        eval_metric='mean_absolute_error',
        path=str(results_dir)
    )
    
    # sample_weight 파라미터 제거 (AutoGluon에서 지원하지 않음)
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=1800,  # 30 minutes
        presets='best_quality',
        num_cpus=6,
        num_gpus=0
        # sample_weight=sample_weight  # 이 줄은 제거해야 함
    )
    
    print("Model training completed!")
    print("Leaderboard:")
    print(predictor.leaderboard(silent=True))
    
    return predictor, feature_cols

# -------------------------------
# Forecasting utilities
# -------------------------------

def build_future_row(menu: str, future_date: pd.Timestamp, history: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    """Construct one-row dataframe of features for a specific menu & date using history."""
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
    
    # short-term trend
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
    
    # Robust rolling features
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
    
    tmp["ma_pos_7"] = masked_mean_last(values, 7)
    tmp["ma_pos_14"] = masked_mean_last(values, 14)
    tmp["ma_pos_28"] = masked_mean_last(values, 28)
    tmp["median_7"] = median_last(values, 7)
    tmp["nonzero_rate_28"] = float((values > 0).mean()) if values.size > 0 else 0.0
    
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
    
    # 정비 기간 관련 특성 추가
    tmp["days_since_maintenance"] = 0
    tmp["days_until_maintenance"] = 0
    
    # 가장 최근 정비일로부터의 일수
    past_maintenance = MAINTENANCE_DATES[MAINTENANCE_DATES < future_date]
    if len(past_maintenance) > 0:
        tmp["days_since_maintenance"] = int((future_date - past_maintenance.max()).days)
    
    # 다음 정비일까지의 일수
    future_maintenance = MAINTENANCE_DATES[MAINTENANCE_DATES > future_date]
    if len(future_maintenance) > 0:
        tmp["days_until_maintenance"] = int((future_maintenance.min() - future_date).days)
    
    # static features
    tmp = add_static_features(tmp, lifecycle_map)
    
    # Clean
    num_cols = tmp.select_dtypes(include=[np.number]).columns
    tmp[num_cols] = tmp[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return tmp

def forecast_7_days(predictor: TabularPredictor, feature_cols: list, test_csv_path: str, lifecycle_map: dict) -> dict:
    """Return dict: {menu: [7 preds]} for one TEST file."""
    print(f"[DEBUG] Loading test file: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    df = parse_date(df)
    
    print(f"[DEBUG] Test file shape: {df.shape}")
    print(f"[DEBUG] Unique menus: {len(df['영업장명_메뉴명'].unique())}")
    
    # Build initial history from provided test file
    history = df[["영업일자", "영업장명_메뉴명", "매출수량"]].copy()
    
    preds_per_menu = {}
    menus = sorted(df["영업장명_메뉴명"].unique())
    last_date = history["영업일자"].max()
    
    # 중간 저장을 위한 파일 경로
    test_id = os.path.splitext(os.path.basename(test_csv_path))[0]
    temp_save_path = OUTPUT_DIR / f"temp_{test_id}_predictions.json"
    
    # 기존 중간 저장 파일이 있으면 로드
    if temp_save_path.exists():
        print(f"[INFO] Loading existing temp file: {temp_save_path}")
        with open(temp_save_path, 'r', encoding='utf-8') as f:
            preds_per_menu = json.load(f)
        print(f"[INFO] Loaded {len(preds_per_menu)} existing predictions")
        print(f"[INFO] Resuming from menu {len(preds_per_menu) + 1}")
    
    print(f"[DEBUG] Processing {len(menus)} menus...")
    
    for i, menu in enumerate(menus):
        # 이미 예측된 메뉴는 스킵
        if menu in preds_per_menu:
            print(f"[DEBUG] Skipping menu {i+1}/{len(menus)}: {menu} (already predicted)")
            continue
            
        print(f"[DEBUG] Processing menu {i+1}/{len(menus)}: {menu}")
        preds = []
        menu_hist_full = history[history["영업장명_메뉴명"] == menu].copy()
        
        for step in range(1, 8):
            try:
                future_date = last_date + timedelta(days=step)
                
                # 정비 기간이면 강제로 0 예측
                if future_date.normalize() in MAINTENANCE_SET:
                    preds.append(0)
                    continue
                
                print(f"[DEBUG] Building features for {menu}, day {step}")
                feat_row = build_future_row(menu, future_date, menu_hist_full, lifecycle_map)
                Xf = feat_row[feature_cols].copy()
                
                print(f"[DEBUG] Predicting for {menu}, day {step}")
                # AutoGluon prediction
                yhat_log = float(predictor.predict(Xf)[0])
                yhat = float(np.expm1(yhat_log))
                
                # 최근 히스토리 기반 보정
                recent = menu_hist_full.sort_values("영업일자")
                recent_tail = recent[recent["영업일자"] < future_date].tail(28)
                ma7 = float(recent_tail["매출수량"].tail(7).mean()) if len(recent_tail) > 0 else 0.0
                
                # 요일 평균(최근 4주)
                dow = future_date.dayofweek
                dow_mask = recent_tail["영업일자"].dt.dayofweek == dow
                dow_avg = float(recent_tail.loc[dow_mask, "매출수량"].mean()) if dow_mask.any() else ma7
                
                # 앙상블
                yhat_final = 0.7 * yhat + 0.2 * dow_avg + 0.1 * ma7
                
                # 하한선(floor) 적용
                keywords = ["콜라", "스프라이트", "아메리카노", "카페라떼", "생수", "맥주", "커피", "라떼"]
                has_floor = any(k in menu for k in keywords)
                floor_base = 0.0 if ma7 <= 0 else 0.15 * ma7
                if has_floor:
                    floor_base = max(floor_base, 1.0)
                yhat_final = max(yhat_final, floor_base)
                
                # 상한선 캡(급등 방지)
                hist_max = float(recent_tail["매출수량"].max()) if len(recent_tail) > 0 else yhat_final
                cap = max(1.0, min(hist_max * 1.2, (ma7 * 3.0 if ma7 > 0 else hist_max * 1.2)))
                yhat_final = min(yhat_final, cap)
                
                # Business constraints
                yhat_final = int(max(0, round(yhat_final)))
                preds.append(yhat_final)
                
            except Exception as e:
                print(f"[ERROR] Error processing {menu}, day {step}: {e}")
                preds.append(0)  # fallback
        
        preds_per_menu[menu] = preds
        print(f"[DEBUG] Completed menu {i+1}/{len(menus)}")
        
        # 중간 저장 (5개 메뉴마다) - 더 자주 저장
        if (i + 1) % 5 == 0:
            print(f"[INFO] Saving temp file after {i+1} menus...")
            with open(temp_save_path, 'w', encoding='utf-8') as f:
                json.dump(preds_per_menu, f, ensure_ascii=False, indent=2)
    
    # 최종 저장
    print(f"[INFO] Saving final temp file for {test_id}...")
    with open(temp_save_path, 'w', encoding='utf-8') as f:
        json.dump(preds_per_menu, f, ensure_ascii=False, indent=2)
    
    return preds_per_menu

# -------------------------------
# Association Rules (기존과 동일)
# -------------------------------

def load_rules_map(csv_path: str, conf_thr=0.6, lift_thr=1.3, topn=3):
    r = pd.read_csv(csv_path)
    r = r[(r["confidence"] >= conf_thr) & (r["lift"] >= lift_thr)]
    rules = {}
    for v, g in r.groupby("venue"):
        m = {}
        for a, gg in g.groupby("antecedent"):
            cons = gg.sort_values(["confidence","lift","support_days"], ascending=False)["consequent"].head(topn).tolist()
            m[a] = cons
        rules[v] = m
    return rules

def adjust_with_rules(pred_df, rules_map, alpha=0.3, min_trigger=1):
    if not isinstance(pred_df, pd.DataFrame) or "영업장명_메뉴명" not in pred_df.columns:
        print("[WARNING] pred_df가 올바른 형태가 아닙니다. 원본 예측을 반환합니다.")
        return pred_df
    
    df = pred_df.copy()
    df["venue"] = df["영업장명_메뉴명"].str.split("_").str[0]
    
    for (day, venue), g in df.groupby(["영업일자","venue"]):
        # 정비 기간 체크 수정 - TEST_XX 형태는 건너뛰기
        try:
            if '+' in day:
                day_part = day.split('+')[0]
                # TEST_XX 형태는 건너뛰기 (예측 데이터)
                if day_part.startswith('TEST_'):
                    pass  # 연관규칙 적용 계속 진행
                else:
                    # 실제 날짜인 경우에만 정비 기간 체크
                    day_date = pd.to_datetime(day_part)
                    if day_date.normalize() in MAINTENANCE_SET:
                        continue
        except Exception as e:
            print(f"[DEBUG] Date parsing skipped for {day}: {e}")
            pass  # 파싱 오류 시 무시하고 계속 진행
        
        preds = g.set_index("영업장명_메뉴명")["매출수량"].astype(float).to_dict()
        vr = rules_map.get(venue, {})
        changed = False
        
        for a, cons_list in vr.items():
            if a in preds and preds[a] >= min_trigger:
                for b in cons_list:
                    if b in preds:
                        newv = max(preds[b], int(np.ceil(alpha * preds[a])))
                        if newv != preds[b]:
                            preds[b] = newv
                            changed = True
        
        if changed:
            for k, v in preds.items():
                df.loc[(df["영업일자"]==day)&(df["영업장명_메뉴명"]==k), "매출수량"] = int(v)
    
    return df[["영업일자","영업장명_메뉴명","매출수량"]]

# -------------------------------
# Submission builder
# -------------------------------

def build_submission(all_test_preds: dict) -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sub = sample.copy()
    
    for test_id, menu_to_preds in all_test_preds.items():
        for day in range(1, 8):
            row_label = f"{test_id}+{day}일"
            if row_label in sub["영업일자"].values:
                ridx = sub.index[sub["영업일자"] == row_label][0]
                for menu, preds in menu_to_preds.items():
                    if menu in sub.columns:
                        sub.at[ridx, menu] = int(preds[day - 1])
    
    # Ensure integers and non-negative
    for c in sub.columns:
        if c == "영업일자":
            continue
        sub[c] = sub[c].fillna(0).astype(float).clip(lower=0).round().astype(int)
    
    return sub

# -------------------------------
# Main execution
# -------------------------------

def main():
    start = time.time()
    
    # 모델 훈련 스킵 옵션
    SKIP_TRAINING = True  # True로 설정하면 훈련 스킵
    
    if not SKIP_TRAINING:
        print("[STEP] Training with AutoGluon ...")
        predictor, feature_cols = train_autogluon_model()
    else:
        print("[INFO] Loading trained model ...")
        model_path = BASE_ROOT / "autogluon_results/autogluon_20250822_113757"
        predictor = TabularPredictor.load(str(model_path))
        
        # feature_cols 하드코딩
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
            "영업장명_메뉴명", "영업장명", "first_sale_month", "peak_month", "is_new_menu", "is_discontinued", "menu_category",
            # 정비 기간 관련
            "is_maintenance", "is_maintenance_m1", "is_maintenance_p1",
            "days_since_maintenance", "days_until_maintenance",
        ]
    
    # 생명주기 맵 생성 (예측에서 사용)
    print("[INFO] Building lifecycle map for prediction ...")
    train_df = pd.read_csv(TRAIN_DIR / "train.csv")
    train_df = parse_date(train_df)
    lifecycle_map = build_lifecycle_maps(train_df)
    
    print("[STEP] Forecasting 7 days for each TEST file ...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    
    for tf in test_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]
        print(f"  - Predicting for {test_id} ...")
        menu_preds = forecast_7_days(predictor, feature_cols, tf, lifecycle_map)
        all_test_preds[test_id] = menu_preds
    
    # 연관규칙 로드
    print("[STEP] Loading association rules ...")
    rules_map = load_rules_map(str(RULE_CSV), conf_thr=0.6, lift_thr=1.3, topn=3)
    
    # 예측 결과를 DataFrame으로 변환
    all_predictions = []
    for test_id, menu_preds in all_test_preds.items():
        for menu, preds in menu_preds.items():
            for day_idx, pred in enumerate(preds, 1):
                all_predictions.append({
                    "영업일자": f"{test_id}+{day_idx}일",
                    "영업장명_메뉴명": menu,
                    "매출수량": int(pred)
                })
    
    pred_df = pd.DataFrame(all_predictions)
    
    # 연관규칙 보정 적용 (수정: 실제로 적용)
    print("[STEP] Applying association rule adjustments ...")
    try:
        pred_df_adj = adjust_with_rules(pred_df, rules_map, alpha=0.3, min_trigger=1)
        print("[INFO] Association rule adjustments applied successfully")
    except Exception as e:
        print(f"[WARNING] Association rule adjustment failed: {e}")
        print("[INFO] Using original predictions without adjustments")
        pred_df_adj = pred_df
    
    # 보정된 결과를 다시 all_test_preds 형태로 변환
    adjusted_test_preds = {}
    for test_id in all_test_preds.keys():
        adjusted_test_preds[test_id] = {}
        for menu in all_test_preds[test_id].keys():
            adjusted_preds = []
            for day_idx in range(1, 8):
                row_label = f"{test_id}+{day_idx}일"
                mask = (pred_df_adj["영업일자"] == row_label) & (pred_df_adj["영업장명_메뉴명"] == menu)
                if mask.any():
                    adjusted_preds.append(pred_df_adj.loc[mask, "매출수량"].iloc[0])
                else:
                    adjusted_preds.append(all_test_preds[test_id][menu][day_idx - 1])
            adjusted_test_preds[test_id][menu] = adjusted_preds
    
    print("[STEP] Building submission ...")
    sub = build_submission(adjusted_test_preds)
    out_path = os.path.join(OUTPUT_DIR, f"submission_autogluon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Elapsed: {(time.time()-start)/60:.1f} min")
    
    # 중간 저장 파일들 정리
    print("[INFO] Cleaning up temp files...")
    for tf in test_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]
        temp_save_path = OUTPUT_DIR / f"temp_{test_id}_predictions.json"
        if temp_save_path.exists():
            temp_save_path.unlink()
            print(f"[INFO] Removed temp file: {temp_save_path}")

if __name__ == "__main__":
    main()
