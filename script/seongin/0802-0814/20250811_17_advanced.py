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
# 🎯 최종 전략 (v1.7): 고급 피처 엔지니어링 + AutoGluon 최적화
# - 피처 강화: 메뉴 카테고리, 영업장 특성, 상호작용 피처 등 대폭 추가
# - Log 변환: np.log1p를 적용하여 타겟 변수 분포 안정화
# - TS 최적화: known_covariates를 TimeSeriesPredictor에 명시적으로 전달
# - 두 Predictor의 결과를 '담화', '미라시아'에 특화된 가중치로 앙상블
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
TABULAR_TIME_LIMIT = 600
TIMESERIES_TIME_LIMIT = 600

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

def is_korean_holiday(date):
    return int(date in holiday_dates)

class AutoGluonEnsemblePredictor:
    def __init__(self, device):
        self.tabular_predictors = []
        self.timeseries_predictor = None
        self.feature_cols = None
        self.device = device
        # TimeSeriesPredictor가 사용할 covariate 목록
        self.known_covariate_names = [
            'day_of_week', 'is_weekend', 'month', 'day',
            'week_of_year', 'season', 'is_holiday', 'year'
        ]

    def create_advanced_features(self, df):
        """
        다양한 도메인 지식 및 시계열 특성을 결합한 고급 피처 생성 함수
        (기존 create_28day_features와 create_features 함수를 통합 및 강화)
        """
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
        
        # 2. 텍스트(메뉴명/영업장명) 기반 도메인 피처
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].str.join('_')
        
        # 메뉴 카테고리
        df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False).astype(int)
        df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료', na=False).astype(int)
        df['주류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|beer|생맥주', na=False, case=False).astype(int)
        df['한식류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥', na=False).astype(int)
        df['양식류'] = df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또|스파게티', na=False).astype(int)
        
        # 영업장 특성 (One-hot encoding)
        for store in ['포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아', '느티나무 셀프BBQ', '라그로타', '연회장', '화담숲카페']:
            df[store] = (df['영업장명'] == store).astype(int)

        # 3. 28일 윈도우 기반 통계 피처 (TabularPredictor용)
        # 이 부분은 prepare_tabular_training_data에서 데이터프레임 단위로 처리
        
        df = df.drop(columns=['영업장명', '메뉴명'])
        return df

    def prepare_tabular_training_data(self, full_train_df):
        """TabularPredictor 학습을 위한 통합 데이터셋 생성"""
        X_list, y_list = [], []
        
        # 전체 데이터에 대해 고급 피처 일괄 생성
        featured_df = self.create_advanced_features(full_train_df.copy())
        
        for menu_name in tqdm(full_train_df['영업장명_메뉴명'].unique(), desc="피처 데이터셋 생성", leave=False):
            menu_df = featured_df[featured_df['영업장명_메뉴명'] == menu_name].sort_values(by='영업일자')
            sales = menu_df['매출수량'].values
            
            if len(sales) < 35: continue

            for i in range(len(sales) - 34):
                # 28일 통계 피처
                input_data = sales[i:i+28]
                features = {
                    'mean_sales_28d': np.mean(input_data),
                    'std_sales_28d': np.std(input_data),
                    'median_sales_28d': np.median(input_data),
                    'last_7day_mean': np.mean(input_data[-7:]),
                    'recent_trend': np.mean(input_data[-7:]) - np.mean(input_data[-14:-7]),
                }
                
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
        """known_covariates를 사용하는 TimeSeriesPredictor 모델 학습"""
        print("\n--- TimeSeriesPredictor 모델 학습 시작 (known_covariates 사용) ---")
        
        # full_train_df에 직접 피처를 생성하여 열을 추가합니다.
        df_with_features = self.create_advanced_features(full_train_df.copy())

        ts_df = TimeSeriesDataFrame.from_data_frame(
            df_with_features,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자"
        )

        self.timeseries_predictor = TimeSeriesPredictor(
            label='매출수량',
            path='autogluon_models/timeseries_advanced',
            prediction_length=7,
            eval_metric='RMSE',
            known_covariates_names=self.known_covariate_names
        ).fit(
            ts_df,
            time_limit=TIMESERIES_TIME_LIMIT,
            presets="best_quality",
        )

    def predict_7days_autogluon_ensemble(self, input_28days_data, last_date, menu_name, context_df):
        """두 AutoGluon Predictor의 예측을 가중 앙상블 (known_covariates 포함)"""
        # 1. TabularPredictor 예측
        # 예측에 필요한 피처 생성 (주의: 28일 통계 피처는 별도 계산)
        temp_df = pd.DataFrame({'영업일자': [last_date], '영업장명_메뉴명': [menu_name]})
        base_features = self.create_advanced_features(temp_df).iloc[0].to_dict()

        stat_features = {
            'mean_sales_28d': np.mean(input_28days_data),
            'std_sales_28d': np.std(input_28days_data),
            'median_sales_28d': np.median(input_28days_data),
            'last_7day_mean': np.mean(input_28days_data[-7:]),
            'recent_trend': np.mean(input_28days_data[-7:]) - np.mean(input_28days_data[-14:-7]),
        }
        base_features.update(stat_features)
        
        X_pred_tabular = pd.DataFrame([base_features])[self.feature_cols]
        tabular_preds = [
            max(0, p.predict(X_pred_tabular).iloc[0]) for p in self.tabular_predictors
        ]

        # 2. TimeSeriesPredictor 예측 (known_covariates 포함)
        future_dates = pd.to_datetime([last_date + pd.Timedelta(days=i) for i in range(1, 8)])
        future_covariates_df = pd.DataFrame({
            '영업일자': future_dates,
            '영업장명_메뉴명': [menu_name] * 7
        })
        future_covariates_featured = self.create_advanced_features(future_covariates_df)

        # known_covariates를 TimeSeriesDataFrame으로 명시적 변환
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_covariates_featured,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자"
        )

        ts_context_df = TimeSeriesDataFrame.from_data_frame(
            context_df, id_column="영업장명_메뉴명", timestamp_column="영업일자"
        )
        ts_preds_raw = self.timeseries_predictor.predict(
            ts_context_df,
            known_covariates=known_covariates
        )
        
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
            
            input_28days_data = context_df['매출수량'].values
            last_date = context_df['영업일자'].max()
            
            # AutoGluon 앙상블 예측
            pred_7days_log = predictor.predict_7days_autogluon_ensemble(input_28days_data, last_date, menu_name, context_df)
            
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
        
        output_filename = 'submission_autogluon_advanced.csv'
        final_submission.to_csv(output_filename, index=False)
        print(f"\n{output_filename} 파일 생성 완료")
    else:
        print("생성된 예측이 없습니다.")

    print("\n=== 🏆 AutoGluon 고급 피처 엔지니어링 모델 ===")
    print(f"✅ 고급 피처 + Log변환 + TS 최적화 적용")
    print(f"✅ '담화', '미라시아'에 특화 가중치 적용: {WEIGHT_CONFIG['special']}")
    print("✅ 대회 규칙을 준수하는 최종 모델 구현 완료")


#     submission_autogluon_advanced.csv 파일 생성 완료

# === 🏆 AutoGluon 고급 피처 엔지니어링 모델 ===
# ✅ 고급 피처 + Log변환 + TS 최적화 적용
# ✅ '담화', '미라시아'에 특화 가중치 적용: {'담화': [0.5, 0.5], '미라시아': [0.4, 0.6]}
# ✅ 대회 규칙을 준수하는 최종 모델 구현 완료
# (_ray_fit pid=25888) [3000]     valid_set's rmse: 0.640075 [repeated 10x across cluster]
# (.venv) 
