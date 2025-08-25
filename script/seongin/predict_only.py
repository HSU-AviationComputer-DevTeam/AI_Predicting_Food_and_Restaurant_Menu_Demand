import os
import glob
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# 경로 설정
BASE_ROOT = Path(r"C:\GitHubRepo\AI_Forecasting_Food_and_Restaurant_Menu_Demand")
BASE_DATA = BASE_ROOT / "data"
TRAIN_DIR = BASE_DATA / "train"
TEST_DIR  = BASE_DATA / "test"
SAMPLE_SUB_PATH = BASE_DATA / "sample_submission.csv"
OUTPUT_DIR = BASE_ROOT / "autogluon_results"

# 상수들 (기존 스크립트에서 복사)
HOLIDAY_DATES = pd.to_datetime([
    "2023-01-01","2023-01-21","2023-01-22","2023-01-23","2023-01-24",
    # ... (전체 휴일 리스트)
])
HOLIDAY_SET = set(pd.DatetimeIndex(HOLIDAY_DATES).normalize())

MAINTENANCE_DATES = pd.to_datetime([
    "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12",
    "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", 
    "2025-03-07", "2025-03-08", "2025-03-09", "2025-03-10"
])
MAINTENANCE_SET = set(pd.DatetimeIndex(MAINTENANCE_DATES).normalize())

# 기존 함수들 복사 (전체 함수들)
def parse_date(df: pd.DataFrame, col: str = "영업일자") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (기존 함수 내용)

def add_group_lag_features(df: pd.DataFrame, value_col: str = "매출수량") -> pd.DataFrame:
    # ... (기존 함수 내용)

def build_lifecycle_maps(train_df: pd.DataFrame):
    # ... (기존 함수 내용)

def add_static_features(df: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    # ... (기존 함수 내용)

def build_future_row(menu: str, future_date: pd.Timestamp, history: pd.DataFrame, lifecycle_map: dict) -> pd.DataFrame:
    # ... (기존 함수 내용)

def forecast_7_days(predictor: TabularPredictor, feature_cols: list, test_csv_path: str, lifecycle_map: dict) -> dict:
    # ... (기존 함수 내용)

def load_rules_map(csv_path: str, conf_thr=0.6, lift_thr=1.3, topn=3):
    # ... (기존 함수 내용)

def adjust_with_rules(pred_df, rules_map, alpha=0.3, min_trigger=1):
    # ... (기존 함수 내용)

def build_submission(all_test_preds: dict) -> pd.DataFrame:
    # ... (기존 함수 내용)

def predict_only():
    start = time.time()
    
    # 저장된 모델 로드
    print("[INFO] Loading trained model ...")
    model_path = BASE_ROOT / "autogluon_results/autogluon_20250822_113757"
    predictor = TabularPredictor.load(str(model_path))
    
    # feature_cols 하드코딩 (기존 스크립트에서 복사)
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
    
    # 생명주기 맵 생성
    print("[INFO] Building lifecycle map for prediction ...")
    train_df = pd.read_csv(TRAIN_DIR / "train.csv")
    train_df = parse_date(train_df)
    lifecycle_map = build_lifecycle_maps(train_df)
    
    # 예측만 실행
    print("[STEP] Forecasting 7 days for each TEST file ...")
    all_test_preds = {}
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
    
    for tf in test_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]
        print(f"  - Predicting for {test_id} ...")
        menu_preds = forecast_7_days(predictor, feature_cols, tf, lifecycle_map)
        all_test_preds[test_id] = menu_preds
    
    # 연관규칙 로드 및 보정
    print("[STEP] Loading association rules ...")
    rule_csv = TRAIN_DIR / "assoc_rules_by_venue.csv"
    rules_map = load_rules_map(str(rule_csv), conf_thr=0.6, lift_thr=1.3, topn=3)
    
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
    
    # 연관규칙 보정 적용
    print("[STEP] Applying association rule adjustments ...")
    pred_df_adj = adjust_with_rules(pred_df, rules_map, alpha=0.3, min_trigger=1)
    
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
    out_path = OUTPUT_DIR / f"submission_autogluon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(out_path, index=False)
    print(f"[DONE] Submission saved: {out_path}")
    print(f"[INFO] Elapsed: {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    predict_only()
