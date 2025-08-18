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
# 🎯 최종 전략 (v1.8): 피처 엔지니어링 고도화 + 계층적 모델링
# - 피처 강화: 다양한 기간(7, 14, 28일)의 통계 피처 및 공휴일 근접도 피처 추가
# - 계층적 모델링: 메뉴 카테고리, 영업장 특성을 정적 피처(static_features)로 활용
# - 모델 안정성: 단순 평균 앙상블로 변경하여 특정 매장 의존성 감소
# - 학습 최적화: 학습 시간 증대 및 예측 시 불필요한 경고 메시지 제거
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

# AutoGluon 학습 설정 (시간 증대)
TABULAR_TIME_LIMIT = 1800  # 30분
TIMESERIES_TIME_LIMIT = 1800 # 30분

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 공휴일 리스트 (2025년 포함)
holiday_dates = pd.to_datetime([
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03",
    "2023-10-09", "2023-12-25", "2024-01-01", "2024-02-09", "2024-02-10",
    "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-05",
    "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
    "2024-09-17", "2024-09-18", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01",
    "2025-05-05", "2025-05-06"
])

def find_nearest_holiday(date, holiday_dates):
    """가장 가까운 공휴일과의 전후 차이를 계산"""
    future_holidays = holiday_dates[holiday_dates >= date]
    past_holidays = holiday_dates[holiday_dates < date]
    
    days_to_next = (future_holidays.min() - date).days if not future_holidays.empty else 365
    days_from_last = (date - past_holidays.max()).days if not past_holidays.empty else 365
    
    return days_to_next, days_from_last

def is_korean_holiday(date):
    return int(date in holiday_dates)

class AutoGluonEnsemblePredictor:
    def __init__(self, device):
        self.tabular_predictors = []
        self.timeseries_predictor = None
        self.feature_cols = None
        self.static_feature_cols = None
        self.device = device
        # TimeSeriesPredictor가 사용할 covariate 목록
        self.known_covariate_names = [
            'day_of_week', 'is_weekend', 'month', 'day',
            'week_of_year', 'season', 'is_holiday', 'year',
            'days_to_next_holiday', 'days_from_last_holiday' # 공휴일 근접도 피처 추가
        ]

    def create_advanced_features(self, df):
        """
        다양한 도메인 지식 및 시계열 특성을 결합한 고급 피처 생성 함수
        (v1.8: 공휴일 근접도 피처 추가, 정적/동적 피처 분리 준비)
        """
        df = df.copy()
        # 1. 날짜 기반 기본 피처
        df['영업일자'] = pd.to_datetime(df['영업일자'])
        df['day_of_week'] = df['영업일자'].dt.weekday
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['영업일자'].dt.month
        df['day'] = df['영업일자'].dt.day
        df['week_of_year'] = df['영업일자'].dt.isocalendar().week.astype(int)
        df['season'] = (df['month'] % 12 + 3) // 3
        df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)
        df['year'] = df['영업일자'].dt.year

        # 공휴일 근접도 피처
        holiday_info = df['영업일자'].apply(lambda x: find_nearest_holiday(x, holiday_dates))
        df['days_to_next_holiday'] = [info[0] for info in holiday_info]
        df['days_from_last_holiday'] = [info[1] for info in holiday_info]
        
        # 2. 텍스트(메뉴명/영업장명) 기반 도메인 피처 -> 정적 피처로 분리
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].str.join('_')
        
        # 메뉴 카테고리 (정적 피처)
        df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False).astype(int)
        df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료', na=False).astype(int)
        df['주류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|beer|생맥주', na=False, case=False).astype(int)
        df['한식류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥', na=False).astype(int)
        df['양식류'] = df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또|스파게티', na=False).astype(int)
        
        # 영업장 특성 (One-hot encoding, 정적 피처)
        store_dummies = pd.get_dummies(df['영업장명'], prefix='store')
        df = pd.concat([df, store_dummies], axis=1)

        if self.static_feature_cols is None:
            self.static_feature_cols = ['분식류', '음료류', '주류', '한식류', '양식류'] + list(store_dummies.columns)

        df = df.drop(columns=['영업장명', '메뉴명'])
        return df

    def prepare_tabular_training_data(self, full_train_df):
        """TabularPredictor 학습을 위한 통합 데이터셋 생성 (v1.8: 다양한 통계 피처 추가)"""
        X_list, y_list = [], []
        
        # 전체 데이터에 대해 고급 피처 일괄 생성
        featured_df = self.create_advanced_features(full_train_df.copy())
        
        for menu_name in tqdm(full_train_df['영업장명_메뉴명'].unique(), desc="피처 데이터셋 생성", leave=False):
            menu_df = featured_df[featured_df['영업장명_메뉴명'] == menu_name].sort_values(by='영업일자')
            sales = menu_df['매출수량'].values
            
            if len(sales) < 35: continue

            for i in range(len(sales) - 34):
                features = {}
                # 다양한 기간(7, 14, 28일)의 통계 피처 생성
                for window in [7, 14, 28]:
                    if i + 28 - window >= 0:
                        input_data = sales[i + 28 - window : i + 28]
                        features[f'mean_sales_{window}d'] = np.mean(input_data)
                        features[f'std_sales_{window}d'] = np.std(input_data)
                        features[f'median_sales_{window}d'] = np.median(input_data)
                        features[f'min_sales_{window}d'] = np.min(input_data)
                        features[f'max_sales_{window}d'] = np.max(input_data)
                
                # 추세 피처
                if i + 28 - 14 >= 0:
                    features['recent_trend_7_vs_14'] = np.mean(sales[i+21:i+28]) - np.mean(sales[i+14:i+21])

                # 기존 피처와 결합
                row_features = menu_df.iloc[i+27].to_dict()
                row_features.update(features)
                X_list.append(row_features)
                
                # 타겟 데이터
                y_list.append(sales[i+28:i+35])

        X_df = pd.DataFrame(X_list).drop(columns=['영업일자', '매출수량'])
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
        """known_covariates와 static_features를 사용하는 TimeSeriesPredictor 모델 학습"""
        print("\n--- TimeSeriesPredictor 모델 학습 시작 (계층적 모델링) ---")
        
        df_with_features = self.create_advanced_features(full_train_df.copy())

        # 정적 피처 추출: 각 아이템별로 변하지 않는 값들
        static_features_df = df_with_features.groupby("영업장명_메뉴명")[self.static_feature_cols].first()

        ts_df = TimeSeriesDataFrame.from_data_frame(
            df_with_features,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자",
            static_features_df=static_features_df
        )

        self.timeseries_predictor = TimeSeriesPredictor(
            label='매출수량',
            path='autogluon_models/timeseries_advanced_v1_8', # 경로 변경
            prediction_length=7,
            eval_metric='RMSE',
            known_covariates_names=self.known_covariate_names,
        ).fit(
            ts_df,
            time_limit=TIMESERIES_TIME_LIMIT,
            presets="best_quality",
            random_seed=42 # 재현성을 위해 추가
        )

    def predict_7days_autogluon_ensemble(self, input_28days_data_df, last_date, menu_name):
        """
        두 AutoGluon Predictor의 예측을 앙상블 (v1.8: 경고 제거, 단순 평균)
        - input_28days_data_df: 매출수량 및 모든 피처가 포함된 28일치 데이터프레임
        """
        # 1. TabularPredictor 예측
        base_features = input_28days_data_df.iloc[-1].to_dict()

        stat_features = {}
        sales_data = input_28days_data_df['매출수량'].values
        for window in [7, 14, 28]:
            input_data = sales_data[-window:]
            stat_features[f'mean_sales_{window}d'] = np.mean(input_data)
            stat_features[f'std_sales_{window}d'] = np.std(input_data)
            stat_features[f'median_sales_{window}d'] = np.median(input_data)
            stat_features[f'min_sales_{window}d'] = np.min(input_data)
            stat_features[f'max_sales_{window}d'] = np.max(input_data)
        
        stat_features['recent_trend_7_vs_14'] = np.mean(sales_data[-7:]) - np.mean(sales_data[-14:-7])

        base_features.update(stat_features)
        
        X_pred_tabular = pd.DataFrame([base_features])[self.feature_cols]
        
        tabular_preds = []
        for p in self.tabular_predictors:
            # get_model_best()를 사용하여 최적 모델을 명시, 경고 제거
            best_model = p.get_model_best()
            pred = p.predict(X_pred_tabular, model=best_model).iloc[0]
            tabular_preds.append(max(0, pred))

        # 2. TimeSeriesPredictor 예측 (static_features 포함)
        future_dates = pd.to_datetime([last_date + pd.Timedelta(days=i) for i in range(1, 8)])
        future_covariates_df = pd.DataFrame({
            '영업일자': future_dates,
            '영업장명_메뉴명': [menu_name] * 7
        })
        future_covariates_featured = self.create_advanced_features(future_covariates_df)

        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_covariates_featured,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자"
        )
        
        # 예측 시에도 static_features가 포함된 context 데이터가 필요
        ts_context_df = TimeSeriesDataFrame.from_data_frame(
            input_28days_data_df,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자",
            static_features_df=input_28days_data_df.groupby("영업장명_메뉴명")[self.static_feature_cols].first()
        )

        ts_preds_raw = self.timeseries_predictor.predict(
            ts_context_df,
            known_covariates=known_covariates
        )
        
        predictions_for_item = ts_preds_raw.loc[menu_name]
        ts_preds = predictions_for_item['mean'].clip(lower=0).values

        # 3. 단순 평균 앙상블
        ensemble_preds = (np.array(tabular_preds) + ts_preds) / 2
        
        return ensemble_preds

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    print("데이터 로드 중...")
    train_df = pd.read_csv('./data/train/train.csv')
    # Log1p 변환
    train_df['매출수량'] = np.log1p(train_df['매출수량'].clip(lower=0))
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
        
        # test 데이터에도 log1p 변환 적용
        test_file_df['매출수량'] = np.log1p(test_file_df['매출수량'].clip(lower=0))
        # 예측에 필요한 피처를 test_file_df 전체에 미리 생성
        test_file_df_featured = predictor.create_advanced_features(test_file_df.copy())
        
        for menu_name in test_file_df['영업장명_메뉴명'].unique():
            # 28일 입력 데이터 구성 (피처가 포함된 데이터프레임에서)
            context_df = test_file_df_featured[test_file_df_featured['영업장명_메뉴명'] == menu_name].iloc[-28:]
            
            if context_df.empty:
                # submission 파일에 해당 메뉴가 없을 경우 대비
                continue
            
            last_date = context_df['영업일자'].max()
            
            # AutoGluon 앙상블 예측 (입력 데이터 형식 변경)
            pred_7days_log = predictor.predict_7days_autogluon_ensemble(context_df, last_date, menu_name)
            
            # 예측값 복원 (expm1)
            pred_7days = np.expm1(pred_7days_log)
            
            # 결과 저장
            for i, pred_val in enumerate(pred_7days):
                all_predictions.append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': max(0, pred_val) # 혹시 모를 음수값 방지
                })

    # --- 제출 파일 생성 ---
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
        final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
        final_submission = final_submission.fillna(0)
        final_submission = final_submission[submission_df.columns]
        
        output_filename = 'submission_autogluon_v1_8.csv' # 버전명 변경
        final_submission.to_csv(output_filename, index=False)
        print(f"\n{output_filename} 파일 생성 완료")
    else:
        print("생성된 예측이 없습니다.")

    print("\n=== 🏆 AutoGluon 고급 피처 엔지니어링 모델 v1.8 ===")
    print(f"✅ 고급 피처(다기간 통계, 공휴일 근접도) + Log변환 + 계층적 TS 모델링 적용")
    print(f"✅ 단순 평균 앙상블 적용")
    print("✅ 대회 규칙을 준수하는 최종 모델 구현 완료")


#     submission_autogluon_advanced.csv 파일 생성 완료

# === 🏆 AutoGluon 고급 피처 엔지니어링 모델 ===
# ✅ 고급 피처 + Log변환 + TS 최적화 적용
# ✅ '담화', '미라시아'에 특화 가중치 적용: {'담화': [0.5, 0.5], '미라시아': [0.4, 0.6]}
# ✅ 대회 규칙을 준수하는 최종 모델 구현 완료
# (_ray_fit pid=25888) [3000]     valid_set's rmse: 0.640075 [repeated 10x across cluster]
# (.venv) 
