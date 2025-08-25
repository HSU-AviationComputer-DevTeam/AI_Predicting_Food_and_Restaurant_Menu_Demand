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
except Exception as e:
    print("[INFO] TensorFlow not found. Please install it:")
    print("pip install tensorflow")
    raise

# Try to import Darts for N-BEATS
try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
except Exception as e:
    print("[INFO] Darts not found. Please install it:")
    print("pip install darts")
    raise

# 경로 설정 (Windows 환경에 맞게 수정)
BASE_DIR = "C:/GitHubRepo/AI_Forecasting_Food_and_Restaurant_Menu_Demand"
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
# 시계열 특화 모델 클래스들
# -------------------------------

class ProphetModel:
    """Prophet 시계열 모델"""
    
    def __init__(self, menu: str):
        self.menu = menu
        self.model = None
        self.is_trained = False
    
    def train(self, train_df: pd.DataFrame):
        """Prophet 모델 훈련"""
        print(f"[PROPHET] Training for {self.menu}")
        
        # Prophet용 데이터 준비
        menu_data = train_df[train_df['영업장명_메뉴명'] == self.menu].copy()
        if len(menu_data) < 30:  # 최소 데이터 필요
            self.is_trained = False
            return
        
        menu_data['ds'] = menu_data['영업일자']
        menu_data['y'] = menu_data['매출수량']
        
        # Prophet 모델 설정
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # 훈련
        self.model.fit(menu_data[['ds', 'y']])
        self.is_trained = True
    
    def predict(self, future_dates: list) -> list:
        """Prophet 예측"""
        if not self.is_trained or self.model is None:
            return [0.0] * len(future_dates)
        
        try:
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = self.model.predict(future_df)
            return forecast['yhat'].values.tolist()
        except Exception as e:
            print(f"[PROPHET] Error predicting for {self.menu}: {e}")
            return [0.0] * len(future_dates)


class LSTMModel:
    """LSTM 시계열 모델"""
    
    def __init__(self, menu: str, sequence_length: int = 28):
        self.menu = menu
        self.sequence_length = sequence_length
        self.model = None
        self.is_trained = False
        self.scaler = None
    
    def prepare_sequences(self, data: np.ndarray):
        """시계열 데이터를 LSTM용 시퀀스로 변환"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, train_df: pd.DataFrame):
        """LSTM 모델 훈련"""
        print(f"[LSTM] Training for {self.menu}")
        
        # 데이터 준비
        menu_data = train_df[train_df['영업장명_메뉴명'] == self.menu].copy()
        if len(menu_data) < self.sequence_length + 10:
            self.is_trained = False
            return
        
        # 시계열 데이터 정렬
        menu_data = menu_data.sort_values('영업일자')
        sales_data = menu_data['매출수량'].values
        
        # 시퀀스 준비
        X, y = self.prepare_sequences(sales_data)
        
        if len(X) < 10:  # 최소 훈련 데이터 필요
            self.is_trained = False
            return
        
        # LSTM 모델 구축
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # 훈련
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
    
    def predict(self, history_data: np.ndarray, steps: int = 7) -> list:
        """LSTM 예측"""
        if not self.is_trained or self.model is None:
            return [0.0] * steps
        
        try:
            predictions = []
            current_sequence = history_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            for _ in range(steps):
                pred = self.model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(max(0, pred))  # 음수 방지
                
                # 시퀀스 업데이트
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred
            
            return predictions
        except Exception as e:
            print(f"[LSTM] Error predicting for {self.menu}: {e}")
            return [0.0] * steps


class NBEATSModel:
    """N-BEATS 시계열 모델"""
    
    def __init__(self, menu: str):
        self.menu = menu
        self.model = None
        self.is_trained = False
    
    def train(self, train_df: pd.DataFrame):
        """N-BEATS 모델 훈련"""
        print(f"[N-BEATS] Training for {self.menu}")
        
        try:
            # Darts TimeSeries로 변환
            menu_data = train_df[train_df['영업장명_메뉴명'] == self.menu].copy()
            if len(menu_data) < 50:  # N-BEATS는 더 많은 데이터 필요
                self.is_trained = False
                return
            
            menu_data = menu_data.sort_values('영업일자')
            series = TimeSeries.from_dataframe(
                menu_data, 
                '영업일자', 
                '매출수량',
                fill_missing_dates=True,
                freq='D'
            )
            
            # N-BEATS 모델 설정
            self.model = NBEATSModel(
                input_chunk_length=28,
                output_chunk_length=7,
                generic_architecture=True,
                num_stacks=10,
                num_blocks=3,
                num_layers=4,
                layer_widths=256
            )
            
            # 훈련
            self.model.fit(series)
            self.is_trained = True
            
        except Exception as e:
            print(f"[N-BEATS] Error training for {self.menu}: {e}")
            self.is_trained = False
    
    def predict(self, history_series: TimeSeries, steps: int = 7) -> list:
        """N-BEATS 예측"""
        if not self.is_trained or self.model is None:
            return [0.0] * steps
        
        try:
            forecast = self.model.predict(n=steps, series=history_series)
            predictions = forecast.values().flatten().tolist()
            return [max(0, p) for p in predictions]  # 음수 방지
        except Exception as e:
            print(f"[N-BEATS] Error predicting for {self.menu}: {e}")
            return [0.0] * steps


# -------------------------------
# 앙상블 모델 관리자
# -------------------------------

class EnsembleManager:
    """시계열 모델 앙상블 관리자"""
    
    def __init__(self):
        self.prophet_models = {}
        self.lstm_models = {}
        self.nbeats_models = {}
        self.ensemble_weights = {
            'prophet': 0.3,
            'lstm': 0.3,
            'nbeats': 0.2,
            'mljar': 0.2
        }
    
    def train_all_models(self, train_df: pd.DataFrame, mljar_models: dict):
        """모든 모델 훈련"""
        print("[ENSEMBLE] Training all time series models...")
        
        menus = train_df['영업장명_메뉴명'].unique()
        
        for i, menu in enumerate(menus):
            print(f"[ENSEMBLE] Training models for {menu} ({i+1}/{len(menus)})")
            
            # Prophet 모델 훈련
            prophet_model = ProphetModel(menu)
            prophet_model.train(train_df)
            self.prophet_models[menu] = prophet_model
            
            # LSTM 모델 훈련
            lstm_model = LSTMModel(menu)
            lstm_model.train(train_df)
            self.lstm_models[menu] = lstm_model
            
            # N-BEATS 모델 훈련
            nbeats_model = NBEATSModel(menu)
            nbeats_model.train(train_df)
            self.nbeats_models[menu] = nbeats_model
    
    def get_dynamic_weights(self, menu: str) -> dict:
        """메뉴별 동적 가중치 계산"""
        base_weights = self.ensemble_weights.copy()
        
        # 메뉴 특성에 따른 가중치 조정
        if 'BBQ' in menu:
            # BBQ 메뉴는 Prophet이 좋을 수 있음 (계절성 강함)
            base_weights['prophet'] = 0.4
            base_weights['lstm'] = 0.2
            base_weights['nbeats'] = 0.2
            base_weights['mljar'] = 0.2
        elif '카페' in menu:
            # 카페 메뉴는 LSTM이 좋을 수 있음 (복잡한 패턴)
            base_weights['prophet'] = 0.2
            base_weights['lstm'] = 0.4
            base_weights['nbeats'] = 0.2
            base_weights['mljar'] = 0.2
        elif '주막' in menu:
            # 주막 메뉴는 N-BEATS가 좋을 수 있음 (트렌드 변화)
            base_weights['prophet'] = 0.2
            base_weights['lstm'] = 0.2
            base_weights['nbeats'] = 0.4
            base_weights['mljar'] = 0.2
        
        return base_weights
    
    def ensemble_predict(self, menu: str, future_dates: list, history_data: pd.DataFrame, mljar_pred: float) -> float:
        """앙상블 예측"""
        weights = self.get_dynamic_weights(menu)
        
        # 각 모델별 예측
        predictions = {}
        
        # Prophet 예측
        if menu in self.prophet_models:
            prophet_preds = self.prophet_models[menu].predict(future_dates)
            predictions['prophet'] = prophet_preds[0] if prophet_preds else 0.0
        else:
            predictions['prophet'] = 0.0
        
        # LSTM 예측
        if menu in self.lstm_models:
            menu_history = history_data[history_data['영업장명_메뉴명'] == menu]['매출수량'].values
            lstm_preds = self.lstm_models[menu].predict(menu_history, steps=1)
            predictions['lstm'] = lstm_preds[0] if lstm_preds else 0.0
        else:
            predictions['lstm'] = 0.0
        
        # N-BEATS 예측
        if menu in self.nbeats_models:
            try:
                menu_history = history_data[history_data['영업장명_메뉴명'] == menu].copy()
                menu_history = menu_history.sort_values('영업일자')
                series = TimeSeries.from_dataframe(
                    menu_history, 
                    '영업일자', 
                    '매출수량',
                    fill_missing_dates=True,
                    freq='D'
                )
                nbeats_preds = self.nbeats_models[menu].predict(series, steps=1)
                predictions['nbeats'] = nbeats_preds[0] if nbeats_preds else 0.0
            except:
                predictions['nbeats'] = 0.0
        else:
            predictions['nbeats'] = 0.0
        
        # MLJAR 예측
        predictions['mljar'] = mljar_pred
        
        # 가중 평균 계산
        final_pred = 0.0
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.25)
            final_pred += weight * pred
        
        return max(0, final_pred)  # 음수 방지


# -------------------------------
# 기존 코드 유지 (수정된 부분만 표시)
# -------------------------------

def parse_date(df: pd.DataFrame, col: str = "영업일자") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

# ... (기존 함수들 유지) ...

def train_ensemble_model() -> tuple:
    """앙상블 모델 훈련"""
    print("[INFO] Loading train data ...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = parse_date(train_df)
    
    # 기존 MLJAR 모델 훈련 (기존 코드에서 가져옴)
    lifecycle_map = build_lifecycle_maps(train_df)
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
    
    # Build sample weights
    venue = train_feat["영업장명"].astype(str)
    base_w = np.where(y <= 0, 0.0, 1.0)
    venue_w = venue.map(SHOP_WEIGHTS).astype(float).fillna(1.0).values
    sample_weight = base_w * venue_w
    
    # Use log1p target for more stable training on counts
    y_train = np.log1p(y.clip(lower=0))
    
    # AutoML settings
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
            "validation_type": "split",
            "train_ratio": 0.90,
            "shuffle": False,
        },
        golden_features=False,
        features_selection=False,
        kmeans_features=False,
        random_state=42,
    )
    
    # Direct models: train 7 horizon-specific models
    print("[INFO] Training AutoML models (Direct 1..7 days) ...")
    mljar_models = {}
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
        venue_w_h = venue_h.map(SHOP_WEIGHTS).astype(float).fillna(1.0).values
        sample_weight_h = base_w_h * venue_w_h
        yh_log = np.log1p(yh.clip(lower=0))
        # separate results path per horizon
        kwargs = dict(base_automl_kwargs)
        kwargs["results_path"] = os.path.join(results_dir, f"h{h}")
        os.makedirs(kwargs["results_path"], exist_ok=True)
        automl_h = AutoML(**kwargs)
        automl_h.fit(Xh, yh_log, sample_weight=sample_weight_h)
        mljar_models[h] = automl_h
    
    # Learn blend weights
    blend_w = {h: (0.70, 0.20) for h in range(1, 8)}
    
    # 앙상블 매니저 초기화 및 훈련
    ensemble_manager = EnsembleManager()
    ensemble_manager.train_all_models(train_df, mljar_models)
    
    return ensemble_manager, mljar_models, lifecycle_map, feature_cols, blend_w


def forecast_7_days_ensemble(ensemble_manager: EnsembleManager, mljar_models: dict, 
                           lifecycle_map: dict, feature_cols: list, 
                           test_csv_path: str, blend_w: dict = None) -> dict:
    """앙상블 예측"""
    df = pd.read_csv(test_csv_path)
    df = parse_date(df)
    
    history = df[["영업일자", "영업장명_메뉴명", "매출수량"]].copy()
    preds_per_menu = {}
    menus = sorted(df["영업장명_메뉴명"].unique())
    last_date = history["영업일자"].max()
    
    if blend_w is None:
        blend_w = {h: (0.70, 0.20) for h in range(1, 8)}
    
    for menu in menus:
        preds = []
        menu_hist_full = history[history["영업장명_메뉴명"] == menu].copy()
        
        for step in range(1, 8):
            future_date = last_date + timedelta(days=step)
            
            # MLJAR 예측 (기존 방식)
            feat_row = build_future_row(menu, future_date, menu_hist_full, lifecycle_map)
            Xf = feat_row[feature_cols].copy()
            yhat_log = float(mljar_models[step].predict(Xf)[0])
            yhat_mljar = float(np.expm1(yhat_log))
            
            # 앙상블 예측
            future_dates = [future_date]
            yhat_ensemble = ensemble_manager.ensemble_predict(
                menu, future_dates, menu_hist_full, yhat_mljar
            )
            
            # 최종 예측 (앙상블 + 기존 보정)
            recent = menu_hist_full.sort_values("영업일자")
            recent_tail = recent[recent["영업일자"] < future_date].tail(28)
            ma7 = float(recent_tail["매출수량"].tail(7).mean()) if len(recent_tail) > 0 else 0.0
            
            dow = future_date.dayofweek
            dow_mask = recent_tail["영업일자"].dt.dayofweek == dow
            dow_avg = float(recent_tail.loc[dow_mask, "매출수량"].mean()) if dow_mask.any() else ma7
            
            w_model, w_dow = blend_w.get(step, (0.70, 0.20))
            w_ma7 = max(0.0, 1.0 - w_model - w_dow)
            
            # 앙상블 결과를 주요 예측으로 사용
            yhat = 0.7 * yhat_ensemble + 0.2 * dow_avg + 0.1 * ma7
            
            # 비즈니스 제약 조건
            keywords = ["콜라", "스프라이트", "아메리카노", "카페라떼", "생수", "맥주", "커피", "라떼"]
            has_floor = any(k in menu for k in keywords)
            floor_base = 0.0 if ma7 <= 0 else 0.15 * ma7
            if has_floor:
                floor_base = max(floor_base, 1.0)
            yhat = max(yhat, floor_base)
            
            hist_max = float(recent_tail["매출수량"].max()) if len(recent_tail) > 0 else yhat
            cap = max(1.0, min(hist_max * 1.2, (ma7 * 3.0 if ma7 > 0 else hist_max * 1.2)))
            yhat = min(yhat, cap)
            
            yhat = int(max(0, round(yhat)))
            preds.append(yhat)
        
        preds_per_menu[menu] = preds
    
    return preds_per_menu


def main():
    start = time.time()
    print("[STEP] Training ensemble models...")
    
    # 앙상블 모델 훈련
    ensemble_manager, mljar_models, lifecycle_map, feature_cols, blend_w = train_ensemble_model()
    
    print("[STEP] Forecasting with ensemble...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    
    for tf in test_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]
        print(f"  - Predicting for {test_id} ...")
        menu_preds = forecast_7_days_ensemble(
            ensemble_manager, mljar_models, lifecycle_map, feature_cols, tf, blend_w
        )
        all_test_preds[test_id] = menu_preds
    
    print("[STEP] Building submission ...")
    sub = build_submission(all_test_preds)
    out_path = os.path.join(OUTPUT_DIR, f"submission_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
