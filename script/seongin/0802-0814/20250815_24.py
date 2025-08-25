import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
# make_scorer는 TimeSeriesPredictor에서 사용하지 않으므로 제거
# from autogluon.core.metrics import make_scorer

# Prophet (옵션): 미설치 시 자동 비활성화
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# Prophet 가중치 (최종 앙상블: final = (1-w)*AG + w*Prophet) - TimeSeriesPredictor에서는 사용하지 않음
# PROPHET_WEIGHT = 0.2
ANALYSIS_PATH = 'autogluon_analysis'
PREDICTION_LENGTH = 7  # 예측 기간


# TimeSeriesPredictor는 내장 sMAPE를 사용하므로 사용자 정의 함수 불필요
# # DACON 대회용 SMAPE 평가 지표 정의
# def smape_metric(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     
#     # 대회 규칙: 실제 매출 수량이 0인 경우는 평가에서 제외
#     mask = y_true != 0
#     
#     # 모든 실제 값이 0인 경우, SMAPE는 0으로 정의
#     if not np.any(mask):
#         return 0.0
#
#     y_true = y_true[mask]
#     y_pred = y_pred[mask]
#     
#     # SMAPE 계산
#     numerator = np.abs(y_pred - y_true)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     
#     # 분모가 0이 되는 경우를 방지 (안전장치로 분모가 0이면 결과는 0)
#     ratio = np.where(denominator == 0, 0, numerator / denominator)
#     
#     return np.mean(ratio) * 100
#
# # AutoGluon용 Scorer 생성
# # 대회의 가중치 SMAPE는 비공개이므로, 일반 SMAPE로 검증 점수를 확인합니다.
# smape_scorer = make_scorer(name='smape',
#                            score_func=smape_metric,
#                            optimum=0,
#                            greater_is_better=False)


# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


# 공휴일 리스트 및 함수
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


def build_prophet_holidays_df():
    hd = pd.DataFrame({
        'holiday': 'kr_holiday',
        'ds': holiday_dates
    })
    return hd

# Prophet을 활용한 피처 생성 함수
def create_prophet_features_and_models(df: pd.DataFrame) -> (pd.DataFrame, dict):
    if Prophet is None:
        return pd.DataFrame(), {}

    all_prophet_features = []
    prophet_models = {}
    unique_menus = df['영업장명_메뉴명'].unique()

    for menu_name in tqdm(unique_menus, desc="Prophet 피처 생성 및 모델 학습", leave=False):
        menu_df = df[df['영업장명_메뉴명'] == menu_name].copy()
        menu_df['영업일자'] = pd.to_datetime(menu_df['영업일자'])
        
        if len(menu_df) < 30:
            continue

        # Prophet 모델 학습
        prophet_df = pd.DataFrame({
            'ds': menu_df['영업일자'],
            'y': np.log1p(menu_df['매출수량'].clip(lower=0))
        })
        
        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=10.0,
                holidays=build_prophet_holidays_df()
            )
            m.fit(prophet_df)
            prophet_models[menu_name] = m  # 학습된 모델 저장

            # 학습 데이터 기간에 대한 예측값 생성
            forecast = m.predict(prophet_df[['ds']])
            forecast = forecast[['ds', 'yhat', 'trend']].rename(columns={
                'ds': '영업일자',
                'yhat': 'prophet_yhat',
                'trend': 'prophet_trend'
            })
            forecast['영업장명_메뉴명'] = menu_name
            all_prophet_features.append(forecast)

        except Exception as e:
            print(f"Prophet 피처 생성 중 '{menu_name}' 메뉴에서 오류 발생: {e}")

    if not all_prophet_features:
        return pd.DataFrame(columns=['영업일자', '영업장명_메뉴명', 'prophet_yhat', 'prophet_trend']), {}

    result_df = pd.concat(all_prophet_features, ignore_index=True)
    result_df['prophet_yhat'] = np.expm1(result_df['prophet_yhat']).clip(lower=0)
    return result_df, prophet_models


# 달력 기반 피처 생성 함수
def create_calendar_features(df, menu_launch_dates=None):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)
    df['year'] = df['영업일자'].dt.year

    # 출시 대비 경과일
    if menu_launch_dates is not None:
        # 예측 시: 미리 계산된 출시일 정보를 사용
        df['min_date'] = df['영업장명_메뉴명'].map(menu_launch_dates)
        df['days_since_launch'] = (df['영업일자'] - df['min_date']).dt.days.fillna(0).astype(int)
        df = df.drop(columns=['min_date'])
    else:
        # 학습 시: 데이터에서 직접 계산
        min_date_by_menu = df.groupby('영업장명_메뉴명')['영업일자'].transform('min')
        df['days_since_launch'] = (df['영업일자'] - min_date_by_menu).dt.days

    # [수정] RecursiveTabular 모델의 오류를 방지하기 위해 명시적 category 변환을 제거합니다.
    # AutoGluon이 내부적으로 타입을 추론하고 처리하도록 맡깁니다.
    # for col in ['day_of_week', 'month', 'year', 'season']:
    #     df[col] = df[col].astype('category')

    return df

def create_advanced_calendar_features(df, menu_launch_dates=None):
    """고급 달력 피처 생성"""
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    
    # 기본 피처
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)
    df['year'] = df['영업일자'].dt.year
    
    # 삼각함수 피처 (계절성 강화)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 월별 특성
    df['is_month_start'] = df['영업일자'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['영업일자'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['영업일자'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['영업일자'].dt.is_quarter_end.astype(int)
    
    # 급여일 근접도 (15일, 25일)
    df['days_to_payday'] = np.minimum(
        abs(df['영업일자'].dt.day - 15),
        abs(df['영업일자'].dt.day - 25)
    )
    df['is_near_payday'] = (df['days_to_payday'] <= 3).astype(int)
    
    # 공휴일 전후
    df['days_to_holiday'] = df['영업일자'].apply(lambda x: 
        min([abs((x - h).days) for h in holiday_dates])
    )
    df['is_holiday_eve'] = (df['days_to_holiday'] == 1).astype(int)
    df['is_holiday_after'] = (df['days_to_holiday'] == 1).astype(int)
    
    # 출시 대비 경과일
    if menu_launch_dates is not None:
        df['min_date'] = df['영업장명_메뉴명'].map(menu_launch_dates)
        df['days_since_launch'] = (df['영업일자'] - df['min_date']).dt.days.fillna(0).astype(int)
        df = df.drop(columns=['min_date'])
    else:
        min_date_by_menu = df.groupby('영업장명_메뉴명')['영업일자'].transform('min')
        df['days_since_launch'] = (df['영업일자'] - min_date_by_menu).dt.days

    return df

def create_advanced_features(df):
    """고급 피처 생성"""
    df = df.copy()
    
    # 삼각함수 피처 (계절성 강화)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 월별 특성
    df['is_month_start'] = df['영업일자'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['영업일자'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['영업일자'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['영업일자'].dt.is_quarter_end.astype(int)
    
    # 급여일 근접도 (15일, 25일)
    df['days_to_payday'] = np.minimum(
        abs(df['영업일자'].dt.day - 15),
        abs(df['영업일자'].dt.day - 25)
    )
    df['is_near_payday'] = (df['days_to_payday'] <= 3).astype(int)
    
    # 공휴일 전후
    df['days_to_holiday'] = df['영업일자'].apply(lambda x: 
        min([abs((x - h).days) for h in holiday_dates])
    )
    df['is_holiday_eve'] = (df['days_to_holiday'] == 1).astype(int)
    df['is_holiday_after'] = (df['days_to_holiday'] == 1).astype(int)
    
    return df

def create_lag_features(df):
    """시계열 Lag 피처 생성"""
    df = df.copy()
    df = df.sort_values(['영업장명_메뉴명', '영업일자'])
    
    # Lag 피처 (1, 7, 14, 21, 28일)
    for lag in [1, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
    
    # Rolling 통계 (7, 14, 28일)
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        df[f'rolling_std_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
        df[f'rolling_min_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].rolling(window, min_periods=1).min().reset_index(0, drop=True)
        df[f'rolling_max_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].rolling(window, min_periods=1).max().reset_index(0, drop=True)
    
    return df


def make_regular_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """각 메뉴별로 빠진 날짜를 채워넣어 시계열을 정규화합니다."""
    regularized_dfs = []
    
    # 전체 데이터의 시작일과 종료일 찾기
    overall_min_date = df['영업일자'].min()
    overall_max_date = df['영업일자'].max()

    for menu_name, menu_df in tqdm(df.groupby('영업장명_메뉴명'), desc="데이터 정규화 (일별)", leave=False):
        # 메뉴별 시작일/종료일이 아닌, 그룹 전체의 시작/종료일을 사용해야
        # 나중에 concat할 때 날짜 인덱스가 일정하게 유지됨
        
        all_dates = pd.date_range(start=menu_df['영업일자'].min(), end=overall_max_date, freq='D')
        
        regular_df = pd.DataFrame({'영업일자': all_dates})
        regular_df['영업장명_메뉴명'] = menu_name
        
        merged_df = pd.merge(regular_df, menu_df, on=['영업일자', '영업장명_메뉴명'], how='left')
        merged_df['매출수량'] = merged_df['매출수량'].fillna(0)
        
        regularized_dfs.append(merged_df)
        
    return pd.concat(regularized_dfs, ignore_index=True)


# 데이터 로드
train_df = pd.read_csv('./data/train/train.csv')
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
submission_df = pd.read_csv('./data/sample_submission.csv')

# [수정] 불규칙한 시계열을 규칙적인 일별 시계열로 변환
train_df = make_regular_time_series(train_df)

# 메뉴별 최초 출시일 미리 계산
menu_launch_dates = train_df.groupby('영업장명_메뉴명')['영업일자'].min()

# 타겟 변수 변환 (로그 변환으로 분포 안정화)
train_df['매출수량_log'] = np.log1p(train_df['매출수량'])

# 이상치 제거 (상위 1% 제거)
q99 = train_df['매출수량'].quantile(0.99)
train_df['매출수량'] = train_df['매출수량'].clip(upper=q99)

# 0값 처리 개선 (매우 작은 값으로 대체)
train_df['매출수량'] = train_df['매출수량'].replace(0, 0.1)

# 로그 변환된 타겟도 업데이트
train_df['매출수량_log'] = np.log1p(train_df['매출수량'])

# 피처 생성 (고급 달력 + Lag + Prophet)
print("📅 고급 달력 기반 피처 생성 중...")
known_features_df = create_advanced_calendar_features(train_df[['영업장명_메뉴명', '영업일자']].drop_duplicates())
train_df = pd.merge(train_df, known_features_df, on=['영업일자', '영업장명_메뉴명'], how='left')

print("📊 Lag 피처 생성 중...")
train_df = create_lag_features(train_df)

prophet_models = {}
if Prophet is not None:
    print("📈 Prophet 피처 생성 중...")
    prophet_features, prophet_models = create_prophet_features_and_models(train_df)
    if not prophet_features.empty:
        train_df = pd.merge(train_df, prophet_features, on=['영업일자', '영업장명_메뉴명'], how='left')

# Prophet 피처에서 발생할 수 있는 NaN 값만 0으로 채웁니다.
if 'prophet_yhat' in train_df.columns:
    train_df['prophet_yhat'] = train_df['prophet_yhat'].fillna(0)
    train_df['prophet_trend'] = train_df['prophet_trend'].fillna(0)

# 추가: 모든 피처의 NaN 값 처리
numeric_columns = train_df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if col != '매출수량':  # 타겟 변수는 제외
        train_df[col] = train_df[col].fillna(0)

# 추가: 데이터 품질 검사
print(f"전체 데이터 크기: {len(train_df)}")
print(f"0이 아닌 매출수량 개수: {len(train_df[train_df['매출수량'] > 0])}")
print(f"NaN 값이 있는 컬럼:")
for col in train_df.columns:
    nan_count = train_df[col].isna().sum()
    if nan_count > 0:
        print(f"  {col}: {nan_count}개")


# test 폴더 내 모든 파일 처리하여 예측
test_paths = sorted(glob.glob('./data/test/*.csv'))
if not test_paths:
    print("test 폴더에 예측할 파일이 없습니다.")
else:
    all_predictions = []
    all_menu_scores = []  # 메뉴별 성능 기록
    unique_menus = train_df['영업장명_메뉴명'].unique()

    for menu_name in tqdm(unique_menus, desc="메뉴별 모델 학습 및 예측"):
        # 1. 메뉴별 데이터 준비
        menu_train_data = train_df[train_df['영업장명_메뉴명'] == menu_name].copy()

        # 학습에 필요한 최소 데이터 확인 (0이 아닌 데이터가 30개 미만이면 건너뛰기)
        if len(menu_train_data.query("매출수량 > 0")) < 30:
            for path in test_paths:
                basename = os.path.basename(path).replace('.csv', '')
                for i in range(7):
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0
                    })
            continue

        # 2. 메뉴별 TimeSeriesDataFrame 생성
        known_covariates_names = [
            'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday', 'year',
            'days_since_launch'
        ]
        if 'prophet_yhat' in menu_train_data.columns:
            known_covariates_names.extend(['prophet_yhat', 'prophet_trend'])

        # 추가: 메뉴별 데이터 품질 검사
        non_zero_count = len(menu_train_data[menu_train_data['매출수량'] > 0])
        print(f"'{menu_name}': 0이 아닌 데이터 {non_zero_count}개")
        
        if non_zero_count < 50:  # 최소 데이터 요구사항 증가
            print(f"⚠️ '{menu_name}' 데이터가 부족하여 건너뜁니다.")
            for path in test_paths:
                basename = os.path.basename(path).replace('.csv', '')
                for i in range(7):
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0
                    })
            continue

        # 추가: NaN 값이 있는 행 제거
        menu_train_data = menu_train_data.dropna(subset=['매출수량'])
        
        # 추가: 피처의 NaN 값 처리
        for col in known_covariates_names:
            if col in menu_train_data.columns:
                menu_train_data[col] = menu_train_data[col].fillna(0)

        train_ts_df = TimeSeriesDataFrame.from_data_frame(
            menu_train_data,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자"
        )

        # 3. 메뉴별 모델 학습
        predictor_path = f'autogluon_timeseries_menu/{menu_name.replace("/", "_").replace(" ", "")}'
        
        try:
            predictor = TimeSeriesPredictor.load(predictor_path)
            print(f"✅ 이미 학습된 '{menu_name}' 모델을 불러왔습니다.")
        except Exception:
            predictor = TimeSeriesPredictor(
                label='매출수량_log',  # 로그 변환된 타겟 사용
                path=predictor_path,
                prediction_length=PREDICTION_LENGTH,
                eval_metric="sMAPE",
                known_covariates_names=known_covariates_names
                # num_val_windows 파라미터 제거
            )
            print(f"🚀 '{menu_name}' 메뉴의 모델 학습을 시작합니다.")
            
            predictor.fit(
                train_ts_df,
                presets="best_quality",  # best_quality로 변경
                time_limit=300,  # 시간 증가
                hyperparameters={
                    "Naive": {}, 
                    "SeasonalNaive": {}, 
                    "DirectTabular": {
                        "max_epochs": 200,  # 대폭 증가
                        "learning_rate": 0.005,  # 조정
                        "dropout": 0.2,  # 증가
                        "hidden_size": 128  # 증가
                    },
                    "ETS": {
                        "trend": "add",
                        "seasonal": "add",
                        "seasonal_periods": 7
                    }, 
                    "Theta": {
                        "seasonal_periods": 7
                    }, 
                    "Chronos": {
                        "model_size": "medium",  # small에서 medium으로
                        "num_samples": 50  # 증가
                    }, 
                    "TemporalFusionTransformer": {
                        "lr": 0.0005,  # 감소
                        "max_epochs": 50,  # 증가
                        "hidden_size": 128,  # 증가
                        "attention_head_size": 8  # 증가
                    },
                },
            )

        # 4. 성능 기록 및 앙상블 가중치 조정
        leaderboard = predictor.leaderboard()
        best_model_entry = leaderboard.iloc[0]
        
        # 모델별 성능에 따른 가중치 계산
        model_weights = {}
        total_score = 0
        for _, row in leaderboard.iterrows():
            score = -row['score_val']  # SMAPE는 낮을수록 좋음
            if score > 0:  # 유효한 점수만
                model_weights[row['model']] = 1 / (score + 0.1)  # 0으로 나누기 방지
                total_score += model_weights[row['model']]
        
        # 정규화
        if total_score > 0:
            for model in model_weights:
                model_weights[model] /= total_score
        
        all_menu_scores.append({
            "menu": menu_name,
            "best_model": best_model_entry["model"],
            "score_val": best_model_entry["score_val"],
            "model_weights": model_weights
        })
        
        print(f"✅ '{menu_name}' Best Model: {best_model_entry['model']} | Validation SMAPE: {-best_model_entry['score_val']:.4f}")
        print(f"   앙상블 가중치: {model_weights}")

        # 5. 메뉴별 순환 예측
        menu_historical_data = menu_train_data.copy()
        for path in test_paths:
            test_day_df = pd.read_csv(path)
            test_day_df['영업일자'] = pd.to_datetime(test_day_df['영업일자'])

            menu_test_day_df = test_day_df[test_day_df['영업장명_메뉴명'] == menu_name]

            menu_historical_data = pd.concat([menu_historical_data, menu_test_day_df], ignore_index=True)
            menu_historical_data = menu_historical_data.drop_duplicates(subset=['영업일자', '영업장명_메뉴명'], keep='last')

            historical_data_regular = make_regular_time_series(menu_historical_data)

            historical_ts_df = TimeSeriesDataFrame.from_data_frame(
                historical_data_regular,
                id_column="영업장명_메뉴명",
                timestamp_column="영업일자"
            )

            last_date = historical_ts_df.index.get_level_values('timestamp').max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)

            future_df = pd.DataFrame({'영업일자': future_dates, '영업장명_메뉴명': menu_name})
            cal_feats = create_calendar_features(future_df, menu_launch_dates=menu_launch_dates)

            if Prophet is not None and menu_name in prophet_models:
                try:
                    m = prophet_models[menu_name]
                    future_prophet_df = pd.DataFrame({'ds': future_df['영업일자']})
                    forecast = m.predict(future_prophet_df)
                    cal_feats['prophet_yhat'] = np.expm1(forecast['yhat'].values).clip(lower=0)
                    cal_feats['prophet_trend'] = forecast['trend'].values
                except Exception:
                    cal_feats['prophet_yhat'] = 0
                    cal_feats['prophet_trend'] = 0
            elif 'prophet_yhat' in train_df.columns:
                cal_feats['prophet_yhat'] = 0
                cal_feats['prophet_trend'] = 0

            future_covariates = TimeSeriesDataFrame.from_data_frame(
                cal_feats,
                id_column="영업장명_메뉴명",
                timestamp_column="영업일자",
            )

            predictions = predictor.predict(historical_ts_df, known_covariates=future_covariates)

            # 로그 변환된 예측을 원래 스케일로 변환
            pred_df_reset = predictions.reset_index()
            pred_df_reset['mean'] = np.expm1(pred_df_reset['mean']).clip(lower=0)

            basename = os.path.basename(path).replace('.csv', '')
            unique_timestamps = sorted(pred_df_reset["timestamp"].unique())

            for i, ts in enumerate(unique_timestamps):
                day_preds = pred_df_reset[pred_df_reset["timestamp"] == ts]
                submission_date_str = f"{basename}+{i+1}일"

                for _, row in day_preds.iterrows():
                    all_predictions.append({
                        '영업일자': submission_date_str,
                        '영업장명_메뉴명': row['item_id'],
                        '매출수량': max(0, row['mean'])
                    })

# 메뉴별 성능 요약 출력
if all_menu_scores:
    print("\n=== 📊 메뉴별 모델 성능 요약 ===")
    scores_df = pd.DataFrame(all_menu_scores)
    scores_df['smape'] = -scores_df['score_val']
    scores_df = scores_df.sort_values('smape', ascending=True)
    
    print(f"총 {len(scores_df)}개 메뉴 처리 완료")
    print(f"평균 SMAPE: {scores_df['smape'].mean():.4f}")
    print(f"최고 성능 메뉴: {scores_df.iloc[0]['menu']} (SMAPE: {scores_df.iloc[0]['smape']:.4f})")
    print(f"최저 성능 메뉴: {scores_df.iloc[-1]['menu']} (SMAPE: {scores_df.iloc[-1]['smape']:.4f})")
    
    # 성능 요약을 파일로 저장
    scores_df.to_csv('menu_performance_summary.csv', index=False)
    print("✅ 메뉴별 성능 요약이 'menu_performance_summary.csv'에 저장되었습니다.")

# 예측 결과를 제출 형식으로 변환
if all_predictions:
    pred_df = pd.DataFrame(all_predictions)
    
    print("\n제출 파일 생성 중...")
    
    submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
    
    final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
    final_submission = final_submission.fillna(0)
    
    # 컬럼 순서를 샘플과 동일하게 맞춤
    final_submission = final_submission[submission_df.columns]
    
    output_filename = 'submission_timeseries_global_model.csv'
    final_submission.to_csv(output_filename, index=False)
    print(f"✅ {output_filename} 파일 생성 완료")

else:
    print("\n생성된 예측이 없습니다.")

print("\n=== 🏆 AutoGluon TimeSeriesPredictor 전역 모델 ===")
print("✅ 모든 메뉴를 하나의 시계열 모델로 학습 및 예측")
print("✅ Prophet 예측을 피처로 활용 (설치 시)")
print("✅ 달력/공휴일 등 미래에 알 수 있는 정보를 known_covariates로 활용")
