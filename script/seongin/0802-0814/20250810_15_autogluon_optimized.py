import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 🎯 최종 전략: AutoGluon 기반 최적화 가중 앙상블
# - 피처 기반 예측: TabularPredictor가 담당 (7개 모델 학습)
# - 시퀀스 기반 예측: TimeSeriesPredictor가 담당 (1개 모델 학습)
# - 두 Predictor의 결과를 '담화', '미라시아'에 특화된 가중치로 앙상블
# - 대회 규칙(28일 고정 윈도우, Data Leakage 방지)을 엄격히 준수
# ==============================================================================

# 가중 앙상블 설정
WEIGHT_CONFIG = {
    # 기본 가중치: [TabularPredictor 가중치, TimeSeriesPredictor 가중치]
    "default": [0.6, 0.4],
    # 업장별 특화 가중치
    "special": {
        "담화": [0.5, 0.5],
        "미라시아": [0.4, 0.6]
    }
}

# AutoGluon 학습 설정
TABULAR_TIME_LIMIT = 180  # 피처 기반 모델 학습 시간 (초)
TIMESERIES_TIME_LIMIT = 300 # 시계열 모델 학습 시간 (초)

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 공휴일 리스트 (도메인 지식)
holiday_dates = pd.to_datetime([
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03",
    "2023-10-09", "2023-12-25", "2024-01-01", "2024-02-09", "2024-02-10",
    "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-05",
    "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
    "2024-09-17", "2024-09-18", "2024-10-03", "2024-10-09", "2024-12-25"
])

def is_korean_holiday(date):
    return int(date in holiday_dates)

class AutoGluonEnsemblePredictor:
    """
    TabularPredictor와 TimeSeriesPredictor를 가중 앙상블하는 모델
    """
    def __init__(self, device):
        self.tabular_predictors = []
        self.timeseries_predictor = None
        self.feature_cols = None
        self.device = device

    def create_28day_features(self, data_28days, last_date, menu_name):
        """28일 데이터로부터 TabularPredictor를 위한 피처 생성"""
        features = {}
        data_28days = np.array(data_28days)
        features['영업장명_메뉴명'] = menu_name # 메뉴명을 피처로 사용

        # 1. 기본 통계량
        features['mean_sales'] = np.mean(data_28days)
        features['std_sales'] = np.std(data_28days)
        features['median_sales'] = np.median(data_28days)
        features['min_sales'] = np.min(data_28days)
        features['max_sales'] = np.max(data_28days)
        
        # 2. 주별 패턴 (4주간)
        for week in range(4):
            week_data = data_28days[week*7:(week+1)*7]
            features[f'week_{week}_mean'] = np.mean(week_data)

        # 3. 최근 경향성
        features['last_7day_mean'] = np.mean(data_28days[-7:])
        features['recent_trend'] = np.mean(data_28days[-7:]) - np.mean(data_28days[-14:-7])

        # 4. 도메인 지식 (추론 시점에서 알 수 있는 정보만)
        # 다음 7일간의 요일, 주말, 공휴일 정보
        for i in range(1, 8):
            pred_date = last_date + pd.Timedelta(days=i)
            features[f'pred_day_{i}_weekday'] = pred_date.weekday()
            features[f'pred_day_{i}_is_weekend'] = 1 if pred_date.weekday() >= 5 else 0
            features[f'pred_day_{i}_is_holiday'] = is_korean_holiday(pred_date)
            
        return features

    def prepare_tabular_training_data(self, full_train_df):
        """TabularPredictor 학습을 위한 통합 데이터셋 생성"""
        X_list, y_list = [], []
        
        for menu_name in tqdm(full_train_df['영업장명_메뉴명'].unique(), desc="피처 데이터셋 생성", leave=False):
            menu_df = full_train_df[full_train_df['영업장명_메뉴명'] == menu_name].sort_values(by='영업일자')
            if len(menu_df) < 35: continue

            sales = menu_df['매출수량'].values
            dates = menu_df['영업일자'].values

            for i in range(len(sales) - 34):
                input_data = sales[i:i+28]
                target_data = sales[i+28:i+35]
                last_date = dates[i+27]
                
                features = self.create_28day_features(input_data, last_date, menu_name)
                X_list.append(features)
                y_list.append(target_data)
        
        X_df = pd.DataFrame(X_list)
        y_df = pd.DataFrame(y_list, columns=[f'target_{i+1}day' for i in range(7)])
        
        if self.feature_cols is None:
            self.feature_cols = X_df.columns
            
        return pd.concat([X_df, y_df], axis=1)

    def train_tabular_predictors(self, train_data):
        """7일 예측을 위한 7개의 TabularPredictor 모델 학습"""
        for day in range(7):
            print(f"\n--- TabularPredictor Day {day+1} 모델 학습 시작 ---")
            label = f'target_{day+1}day'
            predictor = TabularPredictor(
                label=label,
                path=f'autogluon_models/tabular_day_{day+1}',
                problem_type='regression',
                eval_metric='rmse'
            ).fit(
                train_data.drop(columns=[f'target_{i+1}day' for i in range(7) if i != day]),
                time_limit=TABULAR_TIME_LIMIT,
                presets='best_quality'
            )
            self.tabular_predictors.append(predictor)
    
    def train_timeseries_predictor(self, full_train_df):
        """TimeSeriesPredictor 모델 학습"""
        print("\n--- TimeSeriesPredictor 모델 학습 시작 ---")
        ts_df = TimeSeriesDataFrame.from_data_frame(
            full_train_df,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자"
        )
        self.timeseries_predictor = TimeSeriesPredictor(
            label='매출수량',
            path='autogluon_models/timeseries',
            prediction_length=7,
            eval_metric='RMSE',
        ).fit(
            ts_df,
            time_limit=TIMESERIES_TIME_LIMIT,
            presets="best_quality",
        )

    def predict_7days_autogluon_ensemble(self, input_28days_data, last_date, menu_name, context_df):
        """두 AutoGluon Predictor의 예측을 가중 앙상블"""
        # 1. TabularPredictor 예측
        features = self.create_28day_features(input_28days_data, last_date, menu_name)
        X_pred_tabular = pd.DataFrame([features])[self.feature_cols]
        tabular_preds = [
            max(0, p.predict(X_pred_tabular).iloc[0]) for p in self.tabular_predictors
        ]

        # 2. TimeSeriesPredictor 예측
        ts_context_df = TimeSeriesDataFrame.from_data_frame(
            context_df, id_column="영업장명_메뉴명", timestamp_column="영업일자"
        )
        ts_preds_raw = self.timeseries_predictor.predict(ts_context_df)
        
        # BUG FIX: 예측 결과에서 현재 메뉴(menu_name)에 해당하는 값만 정확히 선택
        predictions_for_item = ts_preds_raw.loc[menu_name]
        ts_preds = predictions_for_item['mean'].clip(lower=0).values

        # 3. 가중 앙상블
        store_name = menu_name.split('_')[0]
        weights = WEIGHT_CONFIG['special'].get(store_name, WEIGHT_CONFIG['default'])
        
        ensemble_preds = (np.array(tabular_preds) * weights[0] + ts_preds * weights[1])
        
        return ensemble_preds

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    print("데이터 로드 중...")
    train_df = pd.read_csv('./data/train/train.csv')
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
    train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
    submission_df = pd.read_csv('./data/sample_submission.csv')

    predictor = AutoGluonEnsemblePredictor(device)
    
    # --- 모델 학습 ---
    # 1. TabularPredictor 학습 데이터 준비 및 학습
    tabular_train_data = predictor.prepare_tabular_training_data(train_df)
    predictor.train_tabular_predictors(tabular_train_data)
    
    # 2. TimeSeriesPredictor 학습
    predictor.train_timeseries_predictor(train_df)

    # --- Test 파일별 예측 ---
    print("\n최종 앙상블 예측 생성 중...")
    all_predictions = []
    test_paths = sorted(glob.glob('./data/test/*.csv'))
    
    for path in tqdm(test_paths, desc="Test 파일별 예측"):
        test_file_df = pd.read_csv(path)
        test_file_df['영업일자'] = pd.to_datetime(test_file_df['영업일자'])
        basename = os.path.basename(path).replace('.csv', '')
        
        for menu_name in test_file_df['영업장명_메뉴명'].unique():
            # 28일 입력 데이터 구성 (BUG FIX)
            # 1. 현재 메뉴에 대한 test 데이터만 필터링
            test_menu_df = test_file_df[test_file_df['영업장명_메뉴명'] == menu_name]
            
            # 2. train 데이터에서 가져와야 할 일 수 계산
            days_needed_from_train = 28 - len(test_menu_df)

            # 3. 28일 컨텍스트 데이터 생성
            if days_needed_from_train > 0:
                # train 데이터와 test 데이터를 합쳐 28일 구성
                train_menu_df = train_df[train_df['영업장명_메뉴명'] == menu_name]
                historical_data = train_menu_df.sort_values(by='영업일자').tail(days_needed_from_train)
                context_df = pd.concat([historical_data, test_menu_df])
            else:
                # test 데이터만으로 28일이 채워지는 경우
                context_df = test_menu_df.sort_values(by='영업일자').tail(28)
            
            # TimeSeriesPredictor는 메뉴명이 1개만 있으면 오류 발생하므로, 더미 데이터 추가
            if len(context_df['영업장명_메뉴명'].unique()) < 2:
                dummy_row = context_df.iloc[0].copy()
                dummy_row['영업장명_메뉴명'] = 'dummy_menu_item'
                context_df = pd.concat([context_df, pd.DataFrame([dummy_row])])

            input_28days_data = context_df[context_df['영업장명_메뉴명'] == menu_name]['매출수량'].values
            last_date = context_df[context_df['영업장명_메뉴명'] == menu_name]['영업일자'].max()
            
            # AutoGluon 앙상블 예측
            pred_7days = predictor.predict_7days_autogluon_ensemble(input_28days_data, last_date, menu_name, context_df)
            
            # 결과 저장
            for i, pred_val in enumerate(pred_7days):
                all_predictions.append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred_val
                })

    # --- 제출 파일 생성 ---
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
        final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
        final_submission = final_submission.fillna(0)
        final_submission = final_submission[submission_df.columns]
        
        output_filename = 'submission_autogluon_final_ensemble.csv'
        final_submission.to_csv(output_filename, index=False)
        print(f"\n{output_filename} 파일 생성 완료")
    else:
        print("생성된 예측이 없습니다.")

    print("\n=== 🏆 AutoGluon 기반 최적화 가중 앙상블 모델 ===")
    print(f"✅ TabularPredictor와 TimeSeriesPredictor의 예측을 가중 평균하여 앙상블")
    print(f"✅ '담화', '미라시아'에 특화 가중치 적용: {WEIGHT_CONFIG['special']}")
    print("✅ 대회 규칙을 준수하는 최종 모델 구현 완료")
