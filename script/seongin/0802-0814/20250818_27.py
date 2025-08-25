import os
import sys
import glob
import json
import time
import warnings
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Try to import AutoGluon; if missing, print instruction
try:
    from autogluon.tabular import TabularPredictor
    from autogluon.core.metrics import make_scorer
except Exception as e:
    print("[INFO] AutoGluon not found. Please install it:")
    print("pip install autogluon")
    raise

# 경로 설정 (26번 코드와 동일하게)
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train", "train.csv")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_DIR = BASE_DIR

# 시드 고정 (26번 코드와 동일)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# -------------------------------
# Holiday calendar (26번 코드와 동일)
# -------------------------------

holiday_dates = pd.to_datetime([
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03", 
    "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", 
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15", 
    "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", 
    "2024-10-03", "2024-10-09", "2024-12-25"
])

def is_korean_holiday(date):
    return int(date in holiday_dates)

HOLIDAY_SET = set(pd.DatetimeIndex(holiday_dates).normalize())

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
# Enhanced Utilities for Version 27
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
    
    # Enhanced cyclic encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["weekofyear_sin"] = np.sin(2 * np.pi * df["weekofyear"].astype(float) / 52.0)
    df["weekofyear_cos"] = np.cos(2 * np.pi * df["weekofyear"].astype(float) / 52.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    
    # One-hot encodings for robust seasonality
    for d in range(7):
        df[f"dow_{d}"] = (df["dayofweek"] == d).astype(int)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    
    # Enhanced holiday features
    df["is_holiday"] = df["영업일자"].apply(is_korean_holiday)
    norm_dt = df["영업일자"].dt.normalize()
    df["is_holiday_m1"] = (norm_dt - pd.Timedelta(days=1)).isin(HOLIDAY_SET).astype(int)
    df["is_holiday_p1"] = (norm_dt + pd.Timedelta(days=1)).isin(HOLIDAY_SET).astype(int)
    df["is_holiday_window2"] = ((df["is_holiday"] == 1) | (df["is_holiday_m1"] == 1) | (df["is_holiday_p1"] == 1)).astype(int)
    
    # Additional seasonal features
    df["is_month_start"] = df["day"].isin([1, 2, 3]).astype(int)
    df["is_month_end"] = df["day"].isin([28, 29, 30, 31]).astype(int)
    df["is_quarter_start"] = df["month"].isin([1, 4, 7, 10]).astype(int)
    df["is_quarter_end"] = df["month"].isin([3, 6, 9, 12]).astype(int)
    
    return df


def build_lifecycle_maps(train_df: pd.DataFrame):
    """Enhanced per-menu lifecycle statistics from training data."""
    life = {}
    grouped = train_df[train_df["매출수량"] > 0].groupby("영업장명_메뉴명")
    first_sale = grouped["영업일자"].min()
    last_sale = grouped["영업일자"].max()
    avg_sales = grouped["매출수량"].mean()
    std_sales = grouped["매출수량"].std()
    peak_month = grouped.apply(lambda g: g.groupby(g["영업일자"].dt.month)["매출수량"].sum().idxmax())
    
    # Enhanced lifecycle analysis
    for menu in train_df["영업장명_메뉴명"].unique():
        fs = first_sale.get(menu, pd.NaT)
        ls = last_sale.get(menu, pd.NaT)
        am = float(avg_sales.get(menu, 0.0))
        std = float(std_sales.get(menu, 0.0))
        pm = int(peak_month.get(menu, 6)) if not pd.isna(peak_month.get(menu, np.nan)) else 6
        
        # Enhanced pattern detection
        pattern = "regular"
        if pd.notna(fs) and fs >= pd.Timestamp("2023-06-01"):
            pattern = "new_menu"
        if pd.notna(ls) and ls <= pd.Timestamp("2024-01-31"):
            pattern = "possibly_discontinued"
        
        # Calculate seasonality strength
        menu_data = train_df[train_df["영업장명_메뉴명"] == menu]
        if len(menu_data) > 0:
            monthly_sales = menu_data.groupby(menu_data["영업일자"].dt.month)["매출수량"].mean()
            seasonality_strength = float(monthly_sales.std() / monthly_sales.mean()) if monthly_sales.mean() > 0 else 0.0
        else:
            seasonality_strength = 0.0
        
        life[menu] = {
            "first_sale": fs,
            "last_sale": ls,
            "avg_sales": am,
            "std_sales": std,
            "peak_month": pm,
            "pattern": pattern,
            "seasonality_strength": seasonality_strength,
        }
    return life


def add_group_lag_features(df: pd.DataFrame, value_col: str = "매출수량") -> pd.DataFrame:
    df = df.sort_values(["영업장명_메뉴명", "영업일자"]).copy()
    group = df.groupby("영업장명_메뉴명", group_keys=False)
    
    # Enhanced lag features
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = group[value_col].shift(lag)
    
    # Base shifted series
    s_shift = group[value_col].shift(1)
    s_pos = s_shift.where(s_shift > 0, np.nan)
    
    # Enhanced rolling statistics
    for win in [3, 7, 14, 28]:
        df[f"ma_{win}"] = s_shift.rolling(win, min_periods=1).mean()
        df[f"ma_pos_{win}"] = s_pos.rolling(win, min_periods=1).mean()
        df[f"std_{win}"] = s_shift.rolling(win, min_periods=1).std()
        df[f"median_{win}"] = s_pos.rolling(win, min_periods=1).median()
        df[f"min_{win}"] = s_shift.rolling(win, min_periods=1).min()
        df[f"max_{win}"] = s_shift.rolling(win, min_periods=1).max()
    
    # Enhanced trend features
    df["trend_7"] = (df["ma_7"] / df["ma_7"].shift(7)).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["trend_14"] = (df["ma_14"] / df["ma_14"].shift(14)).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["trend_28"] = (df["ma_28"] / df["ma_28"].shift(28)).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Change rate features
    for lag in [1, 7, 14]:
        df[f"change_rate_{lag}"] = (df[value_col] - df[f"lag_{lag}"]) / (df[f"lag_{lag}"].replace(0, np.nan))
        df[f"change_rate_{lag}"] = df[f"change_rate_{lag}"].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Enhanced day-of-week features
    dow_lags = [f"lag_{k}" for k in [7, 14, 21, 28]]
    df["dow_ma_4"] = df[dow_lags].mean(axis=1)
    df["dow_std_4"] = df[dow_lags].std(axis=1)
    
    # Exponentially weighted means
    df["ewm_7"] = s_shift.ewm(span=7, adjust=False).mean()
    df["ewm_14"] = s_shift.ewm(span=14, adjust=False).mean()
    df["ewm_28"] = s_shift.ewm(span=28, adjust=False).mean()
    
    # Non-zero rate features
    df["nonzero_rate_7"] = s_shift.gt(0).rolling(7, min_periods=1).mean()
    df["nonzero_rate_14"] = s_shift.gt(0).rolling(14, min_periods=1).mean()
    df["nonzero_rate_28"] = s_shift.gt(0).rolling(28, min_periods=1).mean()
    
    # Last sale features
    df["last_pos_date"] = group["영업일자"].apply(lambda s: s.where(df.loc[s.index, value_col] > 0).ffill())
    df["days_since_last_sale"] = (df["영업일자"] - df["last_pos_date"]).dt.days.fillna(9999).astype(int)
    df.drop(columns=["last_pos_date"], inplace=True)
    df["last_nonzero_value"] = group[value_col].apply(lambda s: s.where(s > 0).ffill().shift(1))
    
    # Volatility features
    df["volatility_7"] = df["std_7"] / (df["ma_7"] + 1e-8)
    df["volatility_14"] = df["std_14"] / (df["ma_14"] + 1e-8)
    df["volatility_28"] = df["std_28"] / (df["ma_28"] + 1e-8)
    
    return df


def add_static_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["영업장명"] = df["영업장명_메뉴명"].str.split("_").str[0]
    
    # Enhanced lifecycle encodings
    fs_month = []
    peak_month = []
    new_flag = []
    disc_flag = []
    seasonality_strength = []
    avg_sales = []
    std_sales = []
    
    for m in df["영업장명_메뉴명"].values:
        info = lifecycle_map.get(m, None)
        if info is None:
            fs_month.append(0)
            peak_month.append(6)
            new_flag.append(0)
            disc_flag.append(0)
            seasonality_strength.append(0.0)
            avg_sales.append(0.0)
            std_sales.append(0.0)
        else:
            fs_month.append(0 if pd.isna(info["first_sale"]) else int(info["first_sale"].month))
            peak_month.append(int(info["peak_month"]))
            new_flag.append(1 if info["pattern"] == "new_menu" else 0)
            disc_flag.append(1 if info["pattern"] == "possibly_discontinued" else 0)
            seasonality_strength.append(float(info["seasonality_strength"]))
            avg_sales.append(float(info["avg_sales"]))
            std_sales.append(float(info["std_sales"]))
    
    df["first_sale_month"] = fs_month
    df["peak_month"] = peak_month
    df["is_new_menu"] = new_flag
    df["is_discontinued"] = disc_flag
    df["seasonality_strength"] = seasonality_strength
    df["avg_sales"] = avg_sales
    df["std_sales"] = std_sales
    
    # Enhanced menu category features
    df["is_beverage"] = df["영업장명_메뉴명"].str.contains("콜라|스프라이트|아메리카노|카페라떼|생수|맥주|커피|라떼|음료", regex=True).astype(int)
    df["is_food"] = df["영업장명_메뉴명"].str.contains("밥|국|찌개|구이|볶음|튀김|스테이크|파스타", regex=True).astype(int)
    df["is_dessert"] = df["영업장명_메뉴명"].str.contains("케이크|아이스크림|빵|디저트|후식", regex=True).astype(int)
    
    # Menu name length features
    df["menu_name_length"] = df["영업장명_메뉴명"].str.len()
    df["menu_word_count"] = df["영업장명_메뉴명"].str.count("_") + 1
    
    return df


def make_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    df = parse_date(df)
    df = add_date_features(df)
    df = add_group_lag_features(df)
    df = add_static_features(df, lifecycle_map)
    
    # Clean and handle outliers
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    
    # Robust outlier handling for lag features
    lag_cols = [col for col in df.columns if col.startswith('lag_')]
    for col in lag_cols:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=q99)
    
    # Fill missing values
    df[num_cols] = df[num_cols].fillna(0)
    
    return df


# -------------------------------
# Enhanced Model Training
# -------------------------------

def train_autogluon_model(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> TabularPredictor:
    """Train AutoGluon model for prediction."""
    
    # Prepare data for AutoGluon
    train_data = X.copy()
    train_data['매출수량'] = y
    
    # AutoGluon hyperparameters (26번 코드와 유사)
    hyperparameters = {
        'GBM': [
            {'num_boost_round': 800, 'learning_rate': 0.1, 'max_depth': 6},
            {'num_boost_round': 1200, 'learning_rate': 0.05, 'max_depth': 8}
        ],
        'CAT': [
            {'iterations': 800, 'learning_rate': 0.1, 'depth': 6},
            {'iterations': 1200, 'learning_rate': 0.05, 'depth': 8}
        ],
        'XGB': [
            {'n_estimators': 800, 'learning_rate': 0.1, 'max_depth': 6},
            {'n_estimators': 1200, 'learning_rate': 0.05, 'max_depth': 8}
        ],
        'RF': [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 15}
        ],
        'XT': [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 15}
        ]
    }
    
    # Create predictor
    predictor = TabularPredictor(
        label='매출수량',
        eval_metric='mean_absolute_error',
        path='autogluon_v27_model'
    )
    
    # Fit model
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=600,  # 10 minutes
        presets='medium_quality',
        num_cpus=6,
        num_gpus=0  # CPU only for stability
    )
    
    return predictor


def train_ag_model(train_data: pd.DataFrame, label_col: str, results_path: str, time_limit_sec: int = 900) -> TabularPredictor:
    """Train one AutoGluon predictor under results_path. Assumes any weighting handled upstream (e.g., resampling)."""
    hyperparameters = {
        'GBM': [
            {'num_boost_round': 800, 'learning_rate': 0.1, 'max_depth': 6},
            {'num_boost_round': 1200, 'learning_rate': 0.05, 'max_depth': 8}
        ],
        'CAT': [
            {'iterations': 800, 'learning_rate': 0.1, 'depth': 6},
            {'iterations': 1200, 'learning_rate': 0.05, 'depth': 8}
        ],
        'XGB': [
            {'n_estimators': 800, 'learning_rate': 0.1, 'max_depth': 6},
            {'n_estimators': 1200, 'learning_rate': 0.05, 'max_depth': 8}
        ],
        'RF': [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 15}
        ],
        'XT': [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 15}
        ]
    }
    predictor = TabularPredictor(
        label=label_col,
        eval_metric='mean_absolute_error',
        path=results_path
    )
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=time_limit_sec,
        presets='medium_quality',
        num_cpus=6,
        num_gpus=0,
    )
    return predictor


def train_model() -> tuple:
    print("[INFO] Loading train data ...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)  # 26번 코드와 동일
    train_df = parse_date(train_df)

    lifecycle_map = build_lifecycle_maps(train_df)

    # Build features
    train_feat = make_features(train_df, lifecycle_map)

    # Enhanced feature columns
    feature_cols = [
        # Time features
        "year", "month", "day", "dayofweek", "is_weekend", "is_holiday", "quarter", "weekofyear", "dayofyear",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "weekofyear_sin", "weekofyear_cos", "dayofyear_sin", "dayofyear_cos",
        
        # One-hot encodings
        "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6",
        "month_1","month_2","month_3","month_4","month_5","month_6","month_7","month_8","month_9","month_10","month_11","month_12",
        
        # Holiday features
        "is_holiday_m1","is_holiday_p1","is_holiday_window2",
        "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
        
        # Lag features
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
        
        # Rolling statistics
        "ma_3", "ma_7", "ma_14", "ma_28", 
        "ma_pos_3", "ma_pos_7", "ma_pos_14", "ma_pos_28",
        "std_3", "std_7", "std_14", "std_28",
        "median_3", "median_7", "median_14", "median_28",
        "min_3", "min_7", "min_14", "min_28",
        "max_3", "max_7", "max_14", "max_28",
        
        # Trend features
        "trend_7", "trend_14", "trend_28",
        "change_rate_1", "change_rate_7", "change_rate_14",
        
        # Day-of-week features
        "dow_ma_4", "dow_std_4",
        
        # Exponentially weighted means
        "ewm_7", "ewm_14", "ewm_28",
        
        # Non-zero rate features
        "nonzero_rate_7", "nonzero_rate_14", "nonzero_rate_28",
        
        # Last sale features
        "days_since_last_sale", "last_nonzero_value",
        
        # Volatility features
        "volatility_7", "volatility_14", "volatility_28",
        
        # Static features
        "영업장명_메뉴명", "영업장명", 
        "first_sale_month", "peak_month", "is_new_menu", "is_discontinued",
        "seasonality_strength", "avg_sales", "std_sales",
        "is_beverage", "is_food", "is_dessert",
        "menu_name_length", "menu_word_count",
    ]

    # Drop rows without sufficient history
    train_feat = train_feat.dropna(subset=["lag_1", "lag_7"]).copy()

    X = train_feat[feature_cols]
    y = train_feat["매출수량"].astype(float)

    # Build sample weights
    venue = train_feat["영업장명"].astype(str)
    base_w = np.where(y <= 0, 0.0, 1.0)
    venue_w = venue.map(SHOP_WEIGHTS).astype(float).fillna(1.0).values
    sample_weight = base_w * venue_w

    # Use log1p target for more stable training
    log_target = np.log1p(y.clip(lower=0))

    # -------------------------------
    # Direct models: 1..7-day horizon specific
    # -------------------------------
    print("[INFO] Training AutoGluon models (Direct 1..7 days) with weights ...")
    models_by_h: dict[int, TabularPredictor] = {}
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(OUTPUT_DIR, f"autogluon_v27_direct_{run_tag}")
    os.makedirs(base_results_dir, exist_ok=True)

    for h in range(1, 8):
        # Target at horizon h
        y_target = train_feat.groupby("영업장명_메뉴명")["매출수량"].shift(-h)
        dfh = train_feat.copy()
        dfh["y_target_log"] = np.log1p(dfh["매출수량"].clip(lower=0))  # placeholder to align types
        dfh["y_target_log"] = np.log1p(y_target.clip(lower=0))
        # drop rows with missing lag features or target
        dfh = dfh.dropna(subset=["lag_1", "lag_7", "y_target_log"]).copy()
        Xh = dfh[feature_cols].copy()
        yh_log = dfh["y_target_log"].astype(float)
        # sample weights at horizon (zero target -> weight 0)
        venue_h = dfh["영업장명"].astype(str)
        base_w_h = np.where(np.expm1(yh_log) <= 0, 0.0, 1.0)
        venue_w_h = venue_h.map(SHOP_WEIGHTS).astype(float).fillna(1.0).values
        weight_h = base_w_h * venue_w_h
        # assemble training frame
        train_h = Xh.copy()
        train_h["target_log"] = yh_log
        # weight-driven resampling (approximate weighting)
        # Normalize weights
        w = weight_h.astype(float)
        w_sum = w.sum()
        if w_sum <= 0:
            w = np.ones_like(w)
            w_sum = w.sum()
        w_norm = w / w_sum
        # target sample size ~ original
        n_samples = len(train_h)
        idx = np.random.choice(np.arange(n_samples), size=n_samples, replace=True, p=w_norm)
        train_h = train_h.iloc[idx].reset_index(drop=True)
        # train
        results_path = os.path.join(base_results_dir, f"h{h}")
        os.makedirs(results_path, exist_ok=True)
        models_by_h[h] = train_ag_model(
            train_data=train_h,
            label_col="target_log",
            results_path=results_path,
            time_limit_sec=900,
        )

    # -------------------------------
    # Learn per-horizon blend weights on a recent window from train
    # -------------------------------
    def build_blend_examples(h: int, lookback_days: int = 56):
        records = []  # (y_true, yhat_model, dow_avg, ma7)
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
                yt = g.loc[g["영업일자"] == target_day, "매출수량"]
                if len(yt) == 0:
                    continue
                y_true = float(yt.values[0])
                hist = g[g["영업일자"] <= d][["영업일자", "영업장명_메뉴명", "매출수량"]].copy()
                feat = build_future_row(menu, target_day, hist, lifecycle_map)
                # ensure cols
                miss = set(feature_cols) - set(feat.columns)
                for c in miss:
                    feat[c] = 0.0
                Xb = feat[feature_cols]
                yhat_log = float(models_by_h[h].predict(Xb).iloc[0])
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
        best = (0.7, 0.2, 0.1)
        best_mae = float("inf")
        if not recs:
            return 0.70, 0.20
        for wm in np.arange(0.5, 0.86, grid_step):
            for wd in np.arange(0.1, 0.36, grid_step):
                wma = 1.0 - wm - wd
                if wma < 0:
                    continue
                mae = 0.0
                n = 0
                for y_true, mhat, dhat, ma7 in recs:
                    if y_true <= 0:
                        continue
                    yhat = wm*mhat + wd*dhat + wma*ma7
                    mae += abs(y_true - yhat)
                    n += 1
                if n == 0:
                    continue
                mae /= n
                if mae < best_mae:
                    best_mae = mae
                    best = (wm, wd, wma)
        return best[0], best[1]

    blend_w: dict[int, tuple[float, float]] = {}
    for h in range(1, 8):
        recs = build_blend_examples(h)
        wm, wd = grid_learn_weights(recs)
        blend_w[h] = (float(wm), float(wd))

    # Persist config
    cfg = {
        "feature_cols": feature_cols,
        "use_log1p": True,
        "blend_w": blend_w,
        "models_dir": base_results_dir,
    }
    with open(os.path.join(OUTPUT_DIR, "v27_direct_inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return models_by_h, lifecycle_map, feature_cols, blend_w

# -------------------------------
# Enhanced Forecasting Utilities
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

    # Enhanced lag features
    def get_lag(days: int):
        target_day = future_date - timedelta(days=days)
        v = hist.loc[hist["영업일자"] == target_day, "매출수량"]
        return float(v.values[0]) if len(v) > 0 else np.nan

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        tmp[f"lag_{lag}"] = get_lag(lag)

    # Enhanced rolling statistics
    for win in [3, 7, 14, 28]:
        past_start = future_date - timedelta(days=win)
        mask = (hist["영업일자"] < future_date) & (hist["영업일자"] >= past_start)
        vals = hist.loc[mask, "매출수량"].astype(float).values
        tmp[f"ma_{win}"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        
        # Positive-only mean
        pos_vals = vals[vals > 0]
        tmp[f"ma_pos_{win}"] = float(np.mean(pos_vals)) if len(pos_vals) > 0 else np.nan
        
        # Other statistics
        tmp[f"std_{win}"] = float(np.std(vals)) if len(vals) > 1 else np.nan
        tmp[f"median_{win}"] = float(np.median(pos_vals)) if len(pos_vals) > 0 else np.nan
        tmp[f"min_{win}"] = float(np.min(vals)) if len(vals) > 0 else np.nan
        tmp[f"max_{win}"] = float(np.max(vals)) if len(vals) > 0 else np.nan

    # Enhanced day-of-week features
    l7 = tmp["lag_7"].values[0]
    l14 = tmp["lag_14"].values[0]
    l21 = tmp["lag_21"].values[0]
    l28 = tmp["lag_28"].values[0]
    lag_vals = [v for v in [l7, l14, l21, l28] if not pd.isna(v)]
    tmp["dow_ma_4"] = float(np.mean(lag_vals)) if len(lag_vals) > 0 else 0.0
    tmp["dow_std_4"] = float(np.std(lag_vals)) if len(lag_vals) > 1 else 0.0

    # Enhanced trend features
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
    
    # Add missing trend features
    ma14_now = tmp["ma_14"].values[0]
    prev_start_14 = future_date - timedelta(days=28)
    prev_end_14 = future_date - timedelta(days=15)
    prev_mask_14 = (hist["영업일자"] <= prev_end_14) & (hist["영업일자"] >= prev_start_14)
    prev_vals_14 = hist.loc[prev_mask_14, "매출수량"].astype(float).values
    ma14_prev = float(np.mean(prev_vals_14)) if len(prev_vals_14) > 0 else np.nan
    
    if pd.isna(ma14_prev) or ma14_prev == 0:
        tmp["trend_14"] = 1.0
    else:
        tmp["trend_14"] = float(ma14_now / ma14_prev)
    
    ma28_now = tmp["ma_28"].values[0]
    prev_start_28 = future_date - timedelta(days=56)
    prev_end_28 = future_date - timedelta(days=29)
    prev_mask_28 = (hist["영업일자"] <= prev_end_28) & (hist["영업일자"] >= prev_start_28)
    prev_vals_28 = hist.loc[prev_mask_28, "매출수량"].astype(float).values
    ma28_prev = float(np.mean(prev_vals_28)) if len(prev_vals_28) > 0 else np.nan
    
    if pd.isna(ma28_prev) or ma28_prev == 0:
        tmp["trend_28"] = 1.0
    else:
        tmp["trend_28"] = float(ma28_now / ma28_prev)

    # Enhanced change rate features
    for lag in [1, 7, 14]:
        lag_val = tmp[f"lag_{lag}"].values[0]
        last_val = tmp["lag_1"].values[0]
        if pd.isna(lag_val) or lag_val == 0:
            tmp[f"change_rate_{lag}"] = 0.0
        else:
            tmp[f"change_rate_{lag}"] = float((last_val - lag_val) / lag_val)

    # Enhanced rolling features for inference
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
    
    # Enhanced rolling statistics
    for win in [3, 7, 14, 28]:
        tmp[f"ma_pos_{win}"] = masked_mean_last(values, win)
        tmp[f"median_{win}"] = median_last(values, win)
    
    # Non-zero rate features
    if values.size == 0:
        for win in [7, 14, 28]:
            tmp[f"nonzero_rate_{win}"] = 0.0
    else:
        for win in [7, 14, 28]:
            seg = values[-win:] if len(values) >= win else values
            tmp[f"nonzero_rate_{win}"] = float((seg > 0).mean())
    
    # Exponentially weighted means
    if values.size == 0:
        tmp["ewm_7"] = 0.0
        tmp["ewm_14"] = 0.0
        tmp["ewm_28"] = 0.0
    else:
        s = pd.Series(values)
        tmp["ewm_7"] = float(s.ewm(span=7, adjust=False).mean().iloc[-1])
        tmp["ewm_14"] = float(s.ewm(span=14, adjust=False).mean().iloc[-1])
        tmp["ewm_28"] = float(s.ewm(span=28, adjust=False).mean().iloc[-1])
    
    # Volatility features
    for win in [7, 14, 28]:
        if values.size >= win:
            seg = values[-win:]
            ma = np.mean(seg)
            std = np.std(seg)
            tmp[f"volatility_{win}"] = float(std / (ma + 1e-8))
        else:
            tmp[f"volatility_{win}"] = 0.0
    
    # Last sale features
    hist_pos = history[(history["영업일자"] < future_date) & (history["매출수량"].astype(float) > 0)].sort_values("영업일자")
    if len(hist_pos) == 0:
        tmp["days_since_last_sale"] = int(9999)
        tmp["last_nonzero_value"] = float(0.0)
    else:
        last_row = hist_pos.iloc[-1]
        tmp["days_since_last_sale"] = int((future_date - pd.to_datetime(last_row["영업일자"]).normalize()).days)
        tmp["last_nonzero_value"] = float(last_row["매출수량"])

    # Enhanced static features
    tmp["영업장명"] = menu.split("_")[0]
    info = lifecycle_map.get(menu, None)
    if info is None:
        tmp["first_sale_month"] = 0
        tmp["peak_month"] = 6
        tmp["is_new_menu"] = 0
        tmp["is_discontinued"] = 0
        tmp["seasonality_strength"] = 0.0
        tmp["avg_sales"] = 0.0
        tmp["std_sales"] = 0.0
    else:
        tmp["first_sale_month"] = 0 if pd.isna(info["first_sale"]) else int(info["first_sale"].month)
        tmp["peak_month"] = int(info["peak_month"]) if not pd.isna(info["peak_month"]) else 6
        tmp["is_new_menu"] = 1 if info["pattern"] == "new_menu" else 0
        tmp["is_discontinued"] = 1 if info["pattern"] == "possibly_discontinued" else 0
        tmp["seasonality_strength"] = float(info["seasonality_strength"])
        tmp["avg_sales"] = float(info["avg_sales"])
        tmp["std_sales"] = float(info["std_sales"])
    
    # Enhanced menu category features
    tmp["is_beverage"] = int(any(k in menu for k in ["콜라", "스프라이트", "아메리카노", "카페라떼", "생수", "맥주", "커피", "라떼", "음료"]))
    tmp["is_food"] = int(any(k in menu for k in ["밥", "국", "찌개", "구이", "볶음", "튀김", "스테이크", "파스타"]))
    tmp["is_dessert"] = int(any(k in menu for k in ["케이크", "아이스크림", "빵", "디저트", "후식"]))
    
    # Menu name features
    tmp["menu_name_length"] = len(menu)
    tmp["menu_word_count"] = menu.count("_") + 1

    # Clean
    num_cols = tmp.select_dtypes(include=[np.number]).columns
    tmp[num_cols] = tmp[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return tmp


def predict_ag(predictor: TabularPredictor, X: pd.DataFrame) -> float:
    prediction = predictor.predict(X)
    return float(prediction.iloc[0])


def forecast_7_days(models_or_model, lifecycle_map: dict, feature_cols: list, test_csv_path: str, blend_w: dict | None = None) -> dict:
    """Return dict: {menu: [7 preds]} for one TEST file."""
    df = pd.read_csv(test_csv_path)
    df = parse_date(df)

    # Build initial history from provided test file
    history = df[["영업일자", "영업장명_메뉴명", "매출수량"]].copy()

    preds_per_menu = {}
    menus = sorted(df["영업장명_메뉴명"].unique())

    last_date = history["영업일자"].max()

    is_direct = isinstance(models_or_model, dict)
    models_by_h = models_or_model if is_direct else None
    if blend_w is None:
        blend_w = {h: (0.70, 0.20) for h in range(1, 8)}

    for menu in menus:
        preds = []
        # Use immutable history for direct forecasting
        menu_hist_full = history[history["영업장명_메뉴명"] == menu].copy()
        
        for step in range(1, 8):
            future_date = last_date + timedelta(days=step)
            feat_row = build_future_row(menu, future_date, menu_hist_full, lifecycle_map)
            
            # Ensure all required features are present
            missing_features = set(feature_cols) - set(feat_row.columns)
            for feature in missing_features:
                feat_row[feature] = 0.0
            
            Xf = feat_row[feature_cols].copy()
            
            # AutoGluon prediction (direct or single)
            if is_direct:
                yhat_log = predict_ag(models_by_h[step], Xf)
            else:
                yhat_log = predict_ag(models_or_model, Xf)
            yhat_model = float(np.expm1(yhat_log))
            
            # Enhanced post-processing
            recent = menu_hist_full.sort_values("영업일자")
            recent_tail = recent[recent["영업일자"] < future_date].tail(28)
            ma7 = float(recent_tail["매출수량"].tail(7).mean()) if len(recent_tail) > 0 else 0.0
            
            # Enhanced day-of-week average
            dow = future_date.dayofweek
            dow_mask = recent_tail["영업일자"].dt.dayofweek == dow
            dow_avg = float(recent_tail.loc[dow_mask, "매출수량"].mean()) if dow_mask.any() else ma7
            
            # Per-horizon blend weights
            w_model, w_dow = blend_w.get(step, (0.70, 0.20))
            
            w_ma7 = max(0.0, 1.0 - w_model - w_dow)
            yhat = w_model * yhat_model + w_dow * dow_avg + w_ma7 * ma7
            
            # Enhanced floor and cap logic
            keywords = ["콜라", "스프라이트", "아메리카노", "카페라떼", "생수", "맥주", "커피", "라떼"]
            has_floor = any(k in menu for k in keywords)
            floor_base = 0.0 if ma7 <= 0 else 0.15 * ma7
            if has_floor:
                floor_base = max(floor_base, 1.0)
            yhat = max(yhat, floor_base)
            
            # Enhanced cap logic
            hist_max = float(recent_tail["매출수량"].max()) if len(recent_tail) > 0 else yhat
            cap = max(1.0, min(hist_max * 1.3, (ma7 * 3.5 if ma7 > 0 else hist_max * 1.3)))
            yhat = min(yhat, cap)
            
            # Business constraints (keep float)
            yhat = float(max(0.0, yhat))
            preds.append(yhat)
        
        preds_per_menu[menu] = preds
    return preds_per_menu


def build_submission(all_test_preds: dict) -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    sub = sample.copy()
    
    # Fill predictions
    for test_id, menu_to_preds in all_test_preds.items():
        for day in range(1, 8):
            row_label = f"{test_id}+{day}일"
            if row_label in sub["영업일자"].values:
                ridx = sub.index[sub["영업일자"] == row_label][0]
                for menu, preds in menu_to_preds.items():
                    if menu in sub.columns:
                        sub.at[ridx, menu] = float(preds[day - 1])
    
    # Ensure non-negative floats (no rounding to int)
    for c in sub.columns:
        if c == "영업일자":
            continue
        sub[c] = sub[c].fillna(0).astype(float).clip(lower=0)
    
    return sub


def load_models_and_config(config_path: str | None = None):
    if config_path is None:
        config_path = os.path.join(OUTPUT_DIR, "v27_direct_inference_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    models_dir = cfg["models_dir"]
    feature_cols = cfg["feature_cols"]
    blend_w = {int(h): tuple(v) for h, v in cfg.get("blend_w", {}).items()}
    # Load predictors h1..h7
    models_by_h = {}
    for h in range(1, 8):
        h_path = os.path.join(models_dir, f"h{h}")
        models_by_h[h] = TabularPredictor.load(h_path)
    return models_by_h, feature_cols, blend_w


def inference_only():
    print("[STEP] Inference-only: loading existing models and config ...")
    models_by_h, feature_cols, blend_w = load_models_and_config()
    # Rebuild lifecycle map from train (lightweight)
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
    train_df = parse_date(train_df)
    lifecycle_map = build_lifecycle_maps(train_df)

    print("[STEP] Forecasting 7 days for each TEST file ...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    for tf in tqdm(test_files, desc="테스트 파일별 예측"):
        test_id = os.path.splitext(os.path.basename(tf))[0]
        menu_preds = forecast_7_days(models_by_h, lifecycle_map, feature_cols, tf, blend_w)
        all_test_preds[test_id] = menu_preds

    print("[STEP] Building submission ...")
    sub = build_submission(all_test_preds)
    out_path = os.path.join(OUTPUT_DIR, f"submission_v27_direct_float_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Version 27 inference-only (floats)")


def main():
    start = time.time()
    print("[STEP] Training direct models with weights (Version 27) ...")
    models_by_h, lifecycle_map, feature_cols, blend_w = train_model()

    print("[STEP] Forecasting 7 days for each TEST file ...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    
    for tf in tqdm(test_files, desc="테스트 파일별 예측"):
        test_id = os.path.splitext(os.path.basename(tf))[0]
        menu_preds = forecast_7_days(models_by_h, lifecycle_map, feature_cols, tf, blend_w)
        all_test_preds[test_id] = menu_preds

    print("[STEP] Building submission ...")
    sub = build_submission(all_test_preds)
    out_path = os.path.join(OUTPUT_DIR, f"submission_v27_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Elapsed: {(time.time()-start)/60:.1f} min")
    print(f"[INFO] Version 27 features: Direct AG with {len(feature_cols)} features")


if __name__ == "__main__":
    if "--infer-only" in sys.argv:
        inference_only()
    else:
        main()