import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer

# Prophet (옵션): 미설치 시 자동 비활성화
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# Prophet 가중치 (최종 앙상블: final = (1-w)*AG + w*Prophet)
PROPHET_WEIGHT = 0.2  # 가중치 감소로 AutoGluon 성능 활용
ANALYSIS_PATH = 'autogluon_analysis'
TOP_N_FEATURES = 7 # 피처 선택 시 사용할 상위 피처 개수

# DACON 대회용 SMAPE 평가 지표 정의 (모듈 레벨에서 정의)
def smape_metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 대회 규칙: 실제 매출 수량이 0인 경우는 평가에서 제외
    mask = y_true != 0
    
    # 모든 실제 값이 0인 경우, SMAPE는 0으로 정의
    if not np.any(mask):
        return 0.0

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # SMAPE 계산
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # 분모가 0이 되는 경우를 방지 (안전장치로 분모가 0이면 결과는 0)
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    
    return np.mean(ratio) * 100

# AutoGluon용 Scorer 생성 (모듈 레벨에서 정의)
smape_scorer = make_scorer(name='smape',
                           score_func=smape_metric,
                           optimum=0,
                           greater_is_better=False)

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

# Prophet 보조 예측 캐시: 모델 및 날짜별 예측값
prophet_model_cache = {}
prophet_yhat_by_date_cache = {}

def get_or_fit_prophet_model(train_df_raw: pd.DataFrame, menu_name: str):
    if Prophet is None:
        return None, None
    if menu_name in prophet_model_cache:
        return prophet_model_cache[menu_name]

    series = train_df_raw.loc[train_df_raw['영업장명_메뉴명'] == menu_name, ['영업일자', '매출수량']].copy()
    if series.empty or series['매출수량'].count() < 30:
        return None, None

    series = series.sort_values('영업일자')
    last_train_date = pd.to_datetime(series['영업일자']).max()
    df_p = pd.DataFrame({'ds': pd.to_datetime(series['영업일자']), 'y': np.log1p(series['매출수량'].clip(lower=0))})

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            holidays=build_prophet_holidays_df()
        )
        m.fit(df_p)
        prophet_model_cache[menu_name] = (m, last_train_date)
        return m, last_train_date
    except Exception:
        return None, None

def get_prophet_yhat_for_date(menu_name: str, target_date: pd.Timestamp) -> float | None:
    """
    목표 날짜에 대한 Prophet yhat을 반환. 필요 시 미래 프레임을 확장하여 캐시합니다.
    """
    if Prophet is None:
        return None

    # 모델 준비
    m, last_train_date = prophet_model_cache.get(menu_name, (None, None))
    if m is None:
        return None

    if target_date <= last_train_date:
        return None

    # 캐시 딕셔너리 초기화
    if menu_name not in prophet_yhat_by_date_cache:
        prophet_yhat_by_date_cache[menu_name] = {}

    yhat_map = prophet_yhat_by_date_cache[menu_name]
    if target_date in yhat_map:
        return yhat_map[target_date]

    # 필요한 기간까지 미래 생성 및 예측
    periods = (target_date - last_train_date).days
    if periods <= 0:
        return None
    try:
        future = m.make_future_dataframe(periods=periods, freq='D', include_history=False)
        fcst = m.predict(future)[['ds', 'yhat']]
        # 캐시에 저장
        for _, row in fcst.iterrows():
            ds = pd.to_datetime(row['ds'])
            yhat = max(0.0, float(np.expm1(row['yhat'])))
            yhat_map[ds] = yhat
        return yhat_map.get(target_date)
    except Exception:
        return None

# 피처 생성 함수 (23_best.py 기반)
def create_features(df):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)
    df['year'] = df['영업일자'].dt.year

    # 영업장명과 메뉴명 분리
    if '영업장명_메뉴명' in df.columns:
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].apply(lambda x: '_'.join(x) if x else '')

    # 시간 피처
    sort_keys = ['영업장명_메뉴명', '영업일자'] if '영업장명_메뉴명' in df.columns else ['영업일자']
    df = df.sort_values(sort_keys)
    
    # 출시 대비 경과일 및 월 활성 플래그 생성
    if '영업장명_메뉴명' in df.columns:
        min_date_by_menu = df.groupby('영업장명_메뉴명')['영업일자'].transform('min')
        df['days_since_launch'] = (df['영업일자'] - min_date_by_menu).dt.days
    else:
        df['days_since_launch'] = (df['영업일자'] - df['영업일자'].min()).dt.days

    # is_active_month: 해당 월에 이미 같은 메뉴가 한 번이라도 등장했는지 여부
    df['yyyymm'] = df['영업일자'].dt.to_period('M').astype(str)
    if '영업장명_메뉴명' in df.columns:
        df['is_active_month'] = (df.groupby(['영업장명_메뉴명', 'yyyymm']).cumcount() > 0).astype(int)
    else:
        df['is_active_month'] = (df.groupby(['yyyymm']).cumcount() > 0).astype(int)

    if '매출수량' in df.columns:
        gb_key = '영업장명_메뉴명' if '영업장명_메뉴명' in df.columns else '영업일자'
        gb = df.groupby(gb_key)['매출수량']
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = gb.shift(lag)
        # 7일 rolling만 사용 (14일, 28일은 과적합 위험)
        for window in [7]:
            df[f'rolling_mean_{window}'] = gb.shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = gb.shift(1).rolling(window).std()
    
    # AutoGluon이 날짜 관련 피처를 범주형으로 인식하도록 타입 변경
    df['month'] = df['month'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['year'] = df['year'].astype(str)
    df['season'] = df['season'].astype(str)

    # 불필요한 원본 컬럼 제거
    if '영업장명_메뉴명' in df.columns:
        df = df.drop(columns=['영업장명_메뉴명', '영업장명', '메뉴명'])

    # 내부 계산용 임시 컬럼 제거
    if 'yyyymm' in df.columns:
        df = df.drop(columns=['yyyymm'])

    df = df.fillna(0)
    
    return df

# 데이터 로드
train_df = pd.read_csv('./data/train/train.csv')
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
submission_df = pd.read_csv('./data/sample_submission.csv')

# Prophet 모델 미리 적합(메뉴별)
if Prophet is not None:
    unique_menus_for_prophet = train_df['영업장명_메뉴명'].unique()
    for _menu in tqdm(unique_menus_for_prophet, desc='Prophet 모델 적합', leave=False):
        m, last_d = get_or_fit_prophet_model(train_df, _menu)

# 피처 생성
train_full_featured = create_features(train_df)

# 예측 결과를 저장할 리스트
all_predictions = []
all_menu_scores = [] # 메뉴별 검증 점수 저장 리스트

# 누락 원시 피처 보정 유틸
def align_required_raw_features_for_predict(predictor: TabularPredictor, X: pd.DataFrame) -> pd.DataFrame:
    """
    학습 당시 원시 입력 피처(features_in) 중 현재 X에 누락된 컬럼을 기본값으로 보완합니다.
    """
    try:
        required_cols = list(getattr(predictor._learner.feature_generator, 'features_in'))
    except Exception:
        required_cols = []
    # Known columns 보강
    for col in ['days_since_launch', 'is_active_month']:
        if col not in required_cols:
            required_cols.append(col)
    # 누락 보완
    for col in required_cols:
        if col not in X.columns:
            X[col] = 0
    return X

# test 폴더 내 모든 파일 처리
test_paths = sorted(glob.glob('./data/test/*.csv'))
if not test_paths:
    print("test 폴더에 예측할 파일이 없습니다.")
else:
    unique_menus = train_df['영업장명_메뉴명'].unique()

    for menu_name in tqdm(unique_menus, desc="메뉴별 개별 모델 학습 및 예측"):
        # 1. 메뉴별 데이터 준비
        menu_train_data = train_full_featured[train_df['영업장명_메뉴명'] == menu_name].copy()
        
        if len(menu_train_data) < 30: # 학습에 필요한 최소 데이터 수 (50에서 30으로 감소)
            # 데이터가 부족한 메뉴는 0으로 예측
            for path in test_paths:
                basename = os.path.basename(path).replace('.csv', '')
                for i in range(7):
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0
                    })
            continue
            
        # AutoGluon 학습을 위한 데이터 준비 (영업일자 제외)
        train_data_ag = menu_train_data.drop(columns=['영업일자'])

        # 경로 설정
        predictor_path = f'autogluon_models_menu/{menu_name.replace("/", "_").replace(" ", "")}'
        
        predictor = None
        if os.path.exists(predictor_path):
            try:
                print(f"✅ 이미 학습된 '{menu_name}' 모델 발견. 불러오기를 시작합니다.")
                predictor = TabularPredictor.load(predictor_path)
            except Exception as e:
                print(f" 모델 '{menu_name}' 불러오기 실패 (오류: {e}). 손상된 모델을 삭제하고 새로 학습합니다.")
                shutil.rmtree(predictor_path)
                predictor = None
        
        if predictor is None:
            print(f"🚀 [1단계-탐색] '{menu_name}' 메뉴의 모델 탐색을 시작합니다.")
            
            # 병렬 처리 비활성화로 안정성 확보
            hyperparameters = {
                'GBM': [{}], 'CAT': [{}], 'XGB': [{}], 'RF': [{}], 'XT': [{}]
                # FASTAI 제거 (pickling 오류 원인)
            }
            
            predictor = TabularPredictor(
                label='매출수량', 
                path=predictor_path, 
                problem_type='regression', 
                eval_metric=smape_scorer
            ).fit(
                train_data_ag, 
                hyperparameters=hyperparameters,
                time_limit=300, 
                presets='medium_quality',
                num_bag_folds=3,  # 5에서 3으로 감소
                num_bag_sets=1, 
                ag_args_fit={'num_gpus': 0},  # GPU 비활성화
                ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'}  # 병렬 처리 비활성화
            )

        # 리더보드 및 피처 중요도 분석
        leaderboard = predictor.leaderboard(silent=True)
        best_model_name = leaderboard.iloc[0]['model']
        best_score = leaderboard.iloc[0]['score_val']
        all_menu_scores.append(best_score)
        
        print(f" 메뉴 '{menu_name}' 최종 모델: {best_model_name} (검증 SMAPE: {best_score:.4f})")

        # 2. 순환 예측 (Recursive Forecasting) - 23_best.py의 핵심
        for path in test_paths:
            basename = os.path.basename(path).replace('.csv', '')
            
            # 예측에 사용할 과거 데이터 (train + test 파일의 과거 데이터)
            historical_data = pd.concat([
                train_df[train_df['영업장명_메뉴명'] == menu_name],
                pd.read_csv(path)[lambda x: x['영업장명_메뉴명'] == menu_name]
            ]).copy()
            historical_data['영업일자'] = pd.to_datetime(historical_data['영업일자'])
            
            # 7일 예측
            for i in range(7):
                last_date = historical_data['영업일자'].max()
                next_date = last_date + pd.Timedelta(days=1)
                
                # 예측을 위한 새로운 행 생성
                new_row = pd.DataFrame([{'영업일자': next_date, '영업장명_메뉴명': menu_name, '매출수량': np.nan}])
                
                # 피처 생성을 위해 과거 데이터와 합침
                combined_for_feature = pd.concat([historical_data, new_row], ignore_index=True)
                featured_data = create_features(combined_for_feature)
                
                # 예측할 마지막 행 선택
                X_test = featured_data.tail(1).drop(columns=['영업일자', '매출수량'])

                # AutoGluon 예측
                X_test_aligned = align_required_raw_features_for_predict(predictor, X_test.copy())
                prediction_ag = predictor.predict(X_test_aligned).iloc[0]
                prediction_ag = max(0, prediction_ag)

                # Prophet 보조 예측 (옵션)
                prediction_prophet = get_prophet_yhat_for_date(menu_name, next_date)
                
                # 앙상블
                if prediction_prophet is not None:
                    pred_final = (1 - PROPHET_WEIGHT) * prediction_ag + PROPHET_WEIGHT * prediction_prophet
                else:
                    pred_final = prediction_ag

                # 예측 결과 저장
                all_predictions.append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred_final
                })

                # 다음 예측을 위해 예측값을 포함하여 historical_data 업데이트
                update_row = new_row.copy()
                update_row['매출수량'] = pred_final
                historical_data = pd.concat([historical_data, update_row], ignore_index=True)

# 예측 결과를 제출 형식으로 변환
if all_predictions:
    pred_df = pd.DataFrame(all_predictions)
    
    print("\n제출 파일 생성 중...")
    
    submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
    
    final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
    final_submission = final_submission.fillna(0)
    
    # 컬럼 순서를 샘플과 동일하게 맞춤
    final_submission = final_submission[submission_df.columns]
    
    final_submission.to_csv('submission_autogluon_menu_pipeline_v2.csv', index=False)
    print("✅ submission_autogluon_menu_pipeline_v2.csv 파일 생성 완료")

    # 최종 평균 검증 점수
    if all_menu_scores:
        avg_smape = np.mean(all_menu_scores)
        
        print("\n" + "="*60)
        print(f" 전체 메뉴의 평균 검증 SMAPE 점수: {avg_smape:.4f}")
        print("="*60)
else:
    print("\n생성된 예측이 없습니다.")

print("\n===  AutoGluon(+Prophet 보조 앙상블) 모델 v2 ===")
print("✅ 각 메뉴별 AutoGluon TabularPredictor + Prophet 보조 예측 가중 앙상블")
print("✅ 순환 예측 (Recursive Forecasting)으로 시계열 연속성 고려")
print("✅ Prophet 미설치/데이터 부족 시 자동으로 AutoGluon 단독 예측")
print("✅ 공휴일/주간/연간 시즌성과 추세를 보조로 반영")
