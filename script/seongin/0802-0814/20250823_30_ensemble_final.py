import os
import sys
import glob
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import mljar-supervised
try:
    from supervised.automl import AutoML
except Exception as e:
    print("[INFO] mljar-supervised not found. Please install it:")
    print("pip install mljar-supervised")
    raise

# Try to import Prophet
try:
    from prophet import Prophet
except Exception as e:
    print("[INFO] Prophet not found. Please install it:")
    print("pip install prophet")
    raise

# Try to import TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    print("[INFO] TensorFlow not found. Please install it:")
    print("pip install tensorflow")
    raise

# Try to import Darts for N-BEATS
try:
    from darts import TimeSeries
    from darts.models import NBEATSModel as DartsNBEATSModel
    from darts.utils.missing_values import fill_missing_values
except Exception as e:
    print("[INFO] Darts not found. Please install it:")
    print("pip install darts")
    raise

# 경로 설정
BASE_ROOT = Path(__file__).parent.parent.parent
BASE_DATA = BASE_ROOT / "data"
TRAIN_DIR = BASE_DATA / "train"
TEST_DIR = BASE_DATA / "test"
SAMPLE_SUB_PATH = BASE_DATA / "sample_submission.csv"
OUTPUT_DIR = BASE_ROOT / "ensemble_results"

# 디렉토리 생성
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "화담숲주막": 0.180205,
    "라그로타": 0.214000,
    "느티나무 셀프BBQ": 0.000300,
    "연회장": 0.000300,
    "카페테리아": 0.000300,
    "포레스트릿": 0.000300,
    "화담숲카페": 0.010800,
}

# -------------------------------
# Utilities (기존 코드에서 가져옴)
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
    """Create all features for training/prediction."""
    df = parse_date(df)
    df = add_date_features(df)
    df = add_group_lag_features(df)
    df = add_static_features(df, lifecycle_map)
    return df

# -------------------------------
# 시계열 특화 모델들
# -------------------------------

class ProphetModel:
    """Prophet 시계열 모델"""
    
    def __init__(self):
        self.models = {}
        
    def train(self, train_df: pd.DataFrame, menu: str):
        """Prophet 모델 훈련"""
        try:
            # 메뉴별 데이터 준비
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu].copy()
            if len(menu_data) < 30:  # 최소 데이터 필요
                return None
                
            # Prophet 형식으로 변환
            df_prophet = menu_data[['영업일자', '매출수량']].copy()
            df_prophet.columns = ['ds', 'y']
            df_prophet = df_prophet.sort_values('ds')
            
            # Prophet 모델 생성 및 훈련
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(df_prophet)
            self.models[menu] = model
            print(f"[PROPHET] Trained for {menu}")
            
        except Exception as e:
            print(f"[PROPHET] Error training {menu}: {e}")
            return None
    
    def predict(self, menu: str, future_dates: pd.DatetimeIndex) -> np.ndarray:
        """Prophet 예측"""
        try:
            if menu not in self.models:
                return None
                
            model = self.models[menu]
            
            # 미래 날짜 준비
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future_df)
            
            # 예측값 반환 (음수 방지)
            predictions = np.maximum(0, forecast['yhat'].values)
            return predictions
            
        except Exception as e:
            print(f"[PROPHET] Error predicting {menu}: {e}")
            return None

class LSTMModel:
    """LSTM 딥러닝 시계열 모델"""
    
    def __init__(self, sequence_length=28):  # 30 → 28로 수정
        self.models = {}
        self.sequence_length = sequence_length
        self.scalers = {}
        
    def prepare_sequences(self, data: np.ndarray, sequence_length: int):
        """시계열 데이터를 LSTM 시퀀스로 변환"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, train_df: pd.DataFrame, menu: str):
        """LSTM 모델 훈련"""
        try:
            # 메뉴별 데이터 준비
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu].copy()
            if len(menu_data) < 50:  # 최소 데이터 필요
                return None
                
            # 시계열 데이터 정렬
            menu_data = menu_data.sort_values('영업일자')
            sales_data = menu_data['매출수량'].values
            
            # 데이터가 충분하지 않은 경우 패딩 추가
            if len(sales_data) < self.sequence_length:
                # 0으로 패딩
                padding = np.zeros(self.sequence_length - len(sales_data))
                sales_data = np.concatenate([padding, sales_data])
            
            # 데이터 정규화
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            sales_scaled = scaler.fit_transform(sales_data.reshape(-1, 1)).flatten()
            
            # 시퀀스 준비
            X, y = self.prepare_sequences(sales_scaled, self.sequence_length)
            
            if len(X) < 10:  # 최소 시퀀스 필요
                return None
            
            # LSTM 모델 생성
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            
            # 모델 훈련
            model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
            
            self.models[menu] = model
            self.scalers[menu] = scaler
            print(f"[LSTM] Trained for {menu}")
            
        except Exception as e:
            print(f"[LSTM] Error training {menu}: {e}")
            return None
    
    def predict(self, menu: str, train_df: pd.DataFrame) -> np.ndarray:
        """LSTM 예측"""
        try:
            if menu not in self.models:
                return None
                
            model = self.models[menu]
            scaler = self.scalers[menu]
            
            # 최근 데이터 준비
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu].copy()
            menu_data = menu_data.sort_values('영업일자')
            recent_data = menu_data['매출수량'].tail(self.sequence_length).values
            
            # 데이터가 부족한 경우 패딩 추가
            if len(recent_data) < self.sequence_length:
                padding = np.zeros(self.sequence_length - len(recent_data))
                recent_data = np.concatenate([padding, recent_data])
            
            # 데이터 정규화
            recent_scaled = scaler.transform(recent_data.reshape(-1, 1))
            
            # 7일 예측
            predictions = []
            current_sequence = recent_scaled.reshape(1, self.sequence_length, 1)
            
            for _ in range(7):
                # 다음 값 예측
                next_pred = model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(next_pred)
                
                # 시퀀스 업데이트
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred
            
            # 역정규화 및 음수 방지
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions).flatten()
            predictions = np.maximum(0, predictions)
            
            return predictions
            
        except Exception as e:
            print(f"[LSTM] Error predicting {menu}: {e}")
            return None

class NBEATSModel:
    """N-BEATS 딥러닝 시계열 모델"""
    
    def __init__(self):
        self.models = {}
        
    def train(self, train_df: pd.DataFrame, menu: str):
        """N-BEATS 모델 훈련"""
        try:
            # 메뉴별 데이터 준비
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu].copy()
            if len(menu_data) < 50:  # 최소 데이터 필요
                return None
                
            # 시계열 데이터 정렬
            menu_data = menu_data.sort_values('영업일자')
            
            # Darts TimeSeries 형식으로 변환
            ts = TimeSeries.from_dataframe(
                menu_data, 
                time_col='영업일자', 
                value_cols='매출수량',
                fill_missing_dates=True,
                freq='D'
            )
            
            # 결측값 처리
            ts = fill_missing_values(ts, fill=0.0)
            
            # N-BEATS 모델 생성 및 훈련
            model = DartsNBEATSModel(
                input_chunk_length=30,
                output_chunk_length=7,
                generic_architecture=True,
                num_stacks=10,
                num_blocks=3,
                num_layers=4,
                layer_widths=256,
                random_state=42
            )
            
            model.fit(ts)
            self.models[menu] = model
            print(f"[N-BEATS] Trained for {menu}")
            
        except Exception as e:
            print(f"[N-BEATS] Error training {menu}: {e}")
            return None
    
    def predict(self, menu: str, train_df: pd.DataFrame) -> np.ndarray:
        """N-BEATS 예측"""
        try:
            if menu not in self.models:
                return None
                
            model = self.models[menu]
            
            # 최근 데이터 준비
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu].copy()
            menu_data = menu_data.sort_values('영업일자')
            
            # Darts TimeSeries 형식으로 변환
            ts = TimeSeries.from_dataframe(
                menu_data, 
                time_col='영업일자', 
                value_cols='매출수량',
                fill_missing_dates=True,
                freq='D'
            )
            
            # 결측값 처리
            ts = fill_missing_values(ts, fill=0.0)
            
            # 예측
            forecast = model.predict(n=7, series=ts)
            
            # 예측값 추출 및 음수 방지
            predictions = forecast.values().flatten()
            predictions = np.maximum(0, predictions)
            
            return predictions
            
        except Exception as e:
            print(f"[N-BEATS] Error predicting {menu}: {e}")
            return None

class EnsembleManager:
    """앙상블 모델 관리자"""
    
    def __init__(self):
        self.prophet_model = ProphetModel()
        self.lstm_model = LSTMModel()
        self.nbeats_model = NBEATSModel()
        self.mljar_models = {}
        
    def get_dynamic_weights(self, menu: str) -> dict:
        """메뉴별 동적 가중치 계산"""
        venue = menu.split('_')[0]
        
        # 기본 가중치
        weights = {
            'prophet': 0.25,
            'lstm': 0.25,
            'nbeats': 0.25,
            'mljar': 0.25
        }
        
        # 업장별 특화 가중치
        if 'BBQ' in venue:
            weights['prophet'] = 0.35  # 계절성 강함
            weights['lstm'] = 0.25
            weights['nbeats'] = 0.20
            weights['mljar'] = 0.20
        elif '카페' in venue:
            weights['prophet'] = 0.20
            weights['lstm'] = 0.35  # 복잡한 패턴
            weights['nbeats'] = 0.25
            weights['mljar'] = 0.20
        elif '주막' in venue:
            weights['prophet'] = 0.20
            weights['lstm'] = 0.25
            weights['nbeats'] = 0.35  # 트렌드 변화
            weights['mljar'] = 0.20
        
        return weights
    
    def train_all_models(self, train_df: pd.DataFrame, mljar_models: dict):
        """모든 모델 훈련"""
        print("[ENSEMBLE] Training all time series models...")
        
        menus = sorted(train_df['영업장명_메뉴명'].unique())
        total_menus = len(menus)
        
        for i, menu in enumerate(menus, 1):
            print(f"[ENSEMBLE] Training models for {menu} ({i}/{total_menus})")
            
            # Prophet 훈련
            self.prophet_model.train(train_df, menu)
            
            # LSTM 훈련
            self.lstm_model.train(train_df, menu)
            
            # N-BEATS 훈련
            self.nbeats_model.train(train_df, menu)
        
        # MLJAR 모델 저장
        self.mljar_models = mljar_models
        
        print("[ENSEMBLE] All models trained successfully!")
    
    def predict_ensemble(self, menu: str, train_df: pd.DataFrame, future_dates: pd.DatetimeIndex, mljar_pred: float) -> np.ndarray:
        """앙상블 예측"""
        try:
            # 각 모델별 예측
            prophet_pred = self.prophet_model.predict(menu, future_dates)
            lstm_pred = self.lstm_model.predict(menu, train_df)
            nbeats_pred = self.nbeats_model.predict(menu, train_df)
            
            # 가중치 계산
            weights = self.get_dynamic_weights(menu)
            
            # 예측값 준비
            predictions = []
            
            for i in range(7):
                pred_value = 0.0
                total_weight = 0.0
                
                # Prophet 예측
                if prophet_pred is not None and i < len(prophet_pred):
                    pred_value += weights['prophet'] * prophet_pred[i]
                    total_weight += weights['prophet']
                
                # LSTM 예측
                if lstm_pred is not None and i < len(lstm_pred):
                    pred_value += weights['lstm'] * lstm_pred[i]
                    total_weight += weights['lstm']
                
                # N-BEATS 예측
                if nbeats_pred is not None and i < len(nbeats_pred):
                    pred_value += weights['nbeats'] * nbeats_pred[i]
                    total_weight += weights['nbeats']
                
                # MLJAR 예측 (동일 값 사용)
                pred_value += weights['mljar'] * mljar_pred
                total_weight += weights['mljar']
                
                # 가중 평균 계산
                if total_weight > 0:
                    final_pred = pred_value / total_weight
                else:
                    final_pred = mljar_pred  # 기본값
                
                predictions.append(max(0, final_pred))
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"[ENSEMBLE] Error predicting {menu}: {e}")
            # 오류 시 MLJAR 예측값 반환
            return np.array([mljar_pred] * 7)

# -------------------------------
# MLJAR 모델 훈련 (기존 코드)
# -------------------------------

def train_mljar_model(train_feat: pd.DataFrame, feature_cols: list) -> dict:
    """MLJAR 모델 훈련"""
    print("[MLJAR] Training MLJAR models...")
    
    mljar_models = {}
    menus = sorted(train_feat['영업장명_메뉴명'].unique())
    total_menus = len(menus)
    
    for i, menu in enumerate(menus, 1):
        print(f"[MLJAR] Training for {menu} ({i}/{total_menus})")
        
        # 메뉴별 데이터 준비
        menu_data = train_feat[train_feat['영업장명_메뉴명'] == menu].copy()
        
        if len(menu_data) < 10:
            continue
        
        # 특성과 타겟 분리
        X = menu_data[feature_cols].fillna(0)
        y = np.log1p(menu_data['매출수량'])
        
        # MLJAR 모델 훈련
        model = AutoML(
            results_path=None,
            total_time_limit=60,  # 1분 제한
            mode='Compete',
            eval_metric='mae',
            algorithms=['LightGBM', 'CatBoost'],
            start_random_models=1,
            hill_climbing_steps=2,
            top_models_to_improve=1,
            random_state=42
        )
        
        try:
            model.fit(X, y)
            mljar_models[menu] = model
        except Exception as e:
            print(f"[MLJAR] Error training {menu}: {e}")
            continue
    
    print(f"[MLJAR] Trained {len(mljar_models)} models")
    return mljar_models

# -------------------------------
# 메인 함수
# -------------------------------

def train_ensemble_model() -> tuple:
    """앙상블 모델 훈련"""
    print("[INFO] Loading train data ...")
    train_df = pd.read_csv(TRAIN_DIR / "train.csv")
    print(f"[DEBUG] Train data loaded from: {TRAIN_DIR / 'train.csv'}")
    print(f"[DEBUG] Train data shape: {train_df.shape}")
    train_df = parse_date(train_df)
    
    # 특성 엔지니어링
    lifecycle_map = build_lifecycle_maps(train_df)
    train_feat = make_features(train_df, lifecycle_map)
    
    feature_cols = [
        # time
        "year", "month", "day", "dayofweek", "is_weekend", "is_holiday", "quarter", "weekofyear",
        "dayofyear", "month_sin", "month_cos", "dow_sin", "dow_cos", "weekofyear_sin", "weekofyear_cos",
        "is_holiday_m1", "is_holiday_p1", "is_holiday_window2",
        # lag features
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "ma_7", "ma_14", "ma_28", "ma_pos_7", "ma_pos_14", "ma_pos_28",
        "median_7", "nonzero_rate_28", "ewm_7", "ewm_14", "dow_ma_4",
        "trend_7", "change_rate_7", "days_since_last_sale", "last_nonzero_value",
        # static features
        "first_sale_month", "peak_month", "is_new_menu", "is_discontinued",
        # one-hot encodings
        "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
        "month_1", "month_2", "month_3", "month_4", "month_5", "month_6",
        "month_7", "month_8", "month_9", "month_10", "month_11", "month_12"
    ]
    
    # MLJAR 모델 훈련
    mljar_models = train_mljar_model(train_feat, feature_cols)
    
    # 앙상블 매니저 생성 및 훈련
    ensemble_manager = EnsembleManager()
    ensemble_manager.train_all_models(train_df, mljar_models)
    
    return ensemble_manager, mljar_models, lifecycle_map, feature_cols

def forecast_ensemble(ensemble_manager: EnsembleManager, mljar_models: dict, lifecycle_map: dict, 
                     feature_cols: list, test_csv_path: str) -> dict:
    """앙상블 예측"""
    print(f"[DEBUG] Loading test file: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    df = parse_date(df)
    
    print(f"[DEBUG] Test file shape: {df.shape}")
    print(f"[DEBUG] Unique menus: {len(df['영업장명_메뉴명'].unique())}")
    
    # 초기 히스토리 구축
    history = df[["영업일자", "영업장명_메뉴명", "매출수량"]].copy()
    
    preds_per_menu = {}
    menus = sorted(df["영업장명_메뉴명"].unique())
    last_date = history["영업일자"].max()
    
    print(f"[STEP] Forecasting with ensemble...")
    
    for i, menu in enumerate(menus, 1):
        print(f"[DEBUG] Processing menu {i}/{len(menus)}: {menu}")
        
        try:
            # 미래 날짜 생성
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
            
            # MLJAR 예측 (기본값)
            mljar_pred = 0.0
            if menu in mljar_models:
                # MLJAR 모델을 사용한 실제 예측
                try:
                    # 최근 데이터로 특성 생성
                    menu_history = history[history['영업장명_메뉴명'] == menu].tail(30)
                    if len(menu_history) > 0:
                        # 특성 엔지니어링
                        menu_features = make_features(menu_history, lifecycle_map)
                        if len(menu_features) > 0:
                            # 최신 데이터로 예측
                            latest_features = menu_features[feature_cols].fillna(0).iloc[-1:]
                            mljar_pred_raw = mljar_models[menu].predict(latest_features)[0]
                            mljar_pred = np.expm1(mljar_pred_raw)  # 로그 역변환
                        else:
                            mljar_pred = menu_history['매출수량'].mean()
                    else:
                        mljar_pred = 0.0
                except Exception as e:
                    print(f"[MLJAR] Error predicting {menu}: {e}")
                    mljar_pred = menu_history['매출수량'].mean() if len(menu_history) > 0 else 0.0
            
            # 앙상블 예측
            ensemble_pred = ensemble_manager.predict_ensemble(menu, history, future_dates, mljar_pred)
            
            if ensemble_pred is not None:
                preds_per_menu[menu] = ensemble_pred.tolist()
            else:
                # 오류 시 기본값
                preds_per_menu[menu] = [mljar_pred] * 7
            
            # 중간 저장 (10개 메뉴마다)
            if i % 10 == 0:
                print(f"[INFO] Saving temp file after {i} menus...")
                temp_save_path = OUTPUT_DIR / f"temp_ensemble_predictions.json"
                with open(temp_save_path, 'w', encoding='utf-8') as f:
                    json.dump(preds_per_menu, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"[ERROR] Error processing {menu}: {e}")
            preds_per_menu[menu] = [0.0] * 7
    
    # 최종 저장
    print(f"[INFO] Saving final predictions...")
    final_save_path = OUTPUT_DIR / f"ensemble_predictions.json"
    with open(final_save_path, 'w', encoding='utf-8') as f:
        json.dump(preds_per_menu, f, ensure_ascii=False, indent=2)
    
    return preds_per_menu

def build_submission(predictions_dict: dict, test_files: list) -> pd.DataFrame:
    """제출 파일 생성"""
    print("[STEP] Building submission file...")
    
    rows = []
    for test_file in test_files:
        test_name = Path(test_file).stem
        
        for day in range(1, 8):
            day_col = f"{test_name}+{day}일"
            row = [day_col]
            
            for menu in sorted(predictions_dict.keys()):
                if menu in predictions_dict and len(predictions_dict[menu]) >= day:
                    pred_value = predictions_dict[menu][day - 1]
                else:
                    pred_value = 0.0
                row.append(pred_value)
            
            rows.append(row)
    
    # 컬럼명 생성
    columns = ['영업일자'] + sorted(predictions_dict.keys())
    
    # DataFrame 생성
    submission_df = pd.DataFrame(rows, columns=columns)
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"submission_ensemble_{timestamp}.csv"
    submission_df.to_csv(output_path, index=False)
    
    print(f"[INFO] Submission saved to: {output_path}")
    return submission_df

def main():
    """메인 함수"""
    print("[STEP] Training ensemble models...")
    
    # 앙상블 모델 훈련
    ensemble_manager, mljar_models, lifecycle_map, feature_cols = train_ensemble_model()
    
    # 테스트 파일 목록
    test_files = sorted(glob.glob(str(TEST_DIR / "*.csv")))
    print(f"[INFO] Found {len(test_files)} test files")
    print(f"[DEBUG] Test directory: {TEST_DIR}")
    print(f"[DEBUG] Test files: {[Path(f).name for f in test_files]}")
    
    # 각 테스트 파일별 예측
    all_predictions = {}
    
    for test_file in test_files:
        print(f"[STEP] Processing {Path(test_file).name}...")
        
        # 앙상블 예측
        predictions = forecast_ensemble(
            ensemble_manager, mljar_models, lifecycle_map, feature_cols, test_file
        )
        
        # 예측 결과 병합
        all_predictions.update(predictions)
    
    # 제출 파일 생성
    submission_df = build_submission(all_predictions, test_files)
    
    print("[INFO] Ensemble forecasting completed!")
    print(f"[INFO] Final submission shape: {submission_df.shape}")

if __name__ == "__main__":
    main()
