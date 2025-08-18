import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
# sklearn imports 제거 (사용하지 않음)

# ✅ 추가: GPU 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 첫 번째 GPU 사용
os.environ['CUDA_MEM_FRACTION'] = '0.8'   # GPU 메모리의 80% 사용

# GPU 상태 확인 함수
def check_gpu_status():
    print("=== GPU 상태 확인 ===")
    
    # CUDA 확인
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"   사용 가능한 GPU 수: {torch.cuda.device_count()}")
        else:
            print("❌ PyTorch CUDA 사용 불가")
    except ImportError:
        print("❌ PyTorch 미설치")
    
    # XGBoost GPU 확인
    try:
        import xgboost as xgb
        print(f"✅ XGBoost 설치됨")
    except ImportError:
        print("❌ XGBoost 미설치")
    
    # CatBoost GPU 확인
    try:
        import catboost
        print(f"✅ CatBoost 설치됨")
    except ImportError:
        print("❌ CatBoost 미설치")
    
    print("===================")

# GPU 상태 확인 실행
check_gpu_status()

# Prophet (옵션): 미설치 시 자동 비활성화
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# ✅ 수정: 중복 제거하고 하나로 통일
PROPHET_WEIGHT = 0.15  # AutoGluon에 더 높은 가중치
ANALYSIS_PATH = 'autogluon_analysis'
TOP_N_FEATURES = 10  # 더 많은 피처 선택

# ✅ 수정: 내장 메트릭 사용으로 PicklingError 해결
# DACON 대회용 SMAPE 평가 지표 정의 (별도 계산용)
def smape_metric(y_true, y_pred):
    """DACON 대회용 SMAPE 평가 지표 (별도 계산용)"""
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

# AutoGluon용 Scorer 생성
# 대회의 가중치 SMAPE는 비공개이므로, 일반 SMAPE로 검증 점수를 확인합니다.
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

    # ✅ 수정: 훈련 데이터만 사용
    series = train_df_raw.loc[train_df_raw['영업장명_메뉴명'] == menu_name, ['영업일자', '매출수량']].copy()
    if series.empty or series['매출수량'].count() < 25:
        return None, None

    series = series.sort_values('영업일자')
    last_train_date = pd.to_datetime(series['영업일자']).max()
    df_p = pd.DataFrame({'ds': pd.to_datetime(series['영업일자']), 'y': np.log1p(series['매출수량'].clip(lower=0))})

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # 더 부드러운 추세
            seasonality_prior_scale=5.0,   # 계절성 강도 조절
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
        # 모델이 없다면 학습 시도 (train_df는 전역 범위에서 접근)
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


# 피처 생성 함수 (계절별 메뉴 특성 추가)
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

    # is_active_month
    df['yyyymm'] = df['영업일자'].dt.to_period('M').astype(str)
    if '영업장명_메뉴명' in df.columns:
        df['is_active_month'] = (df.groupby(['영업장명_메뉴명', 'yyyymm']).cumcount() > 0).astype(int)
    else:
        df['is_active_month'] = (df.groupby(['yyyymm']).cumcount() > 0).astype(int)

    if '매출수량' in df.columns:
        gb_key = '영업장명_메뉴명' if '영업장명_메뉴명' in df.columns else '영업일자'
        gb = df.groupby(gb_key)['매출수량']
        
        # 개선된 Lag 피처
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f'lag_{lag}'] = gb.shift(lag)
        
        # 개선된 Rolling 피처
        for window in [7, 14, 28]:
            df[f'rolling_mean_{window}'] = gb.shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = gb.shift(1).rolling(window).std()
            df[f'rolling_min_{window}'] = gb.shift(1).rolling(window).min()
            df[f'rolling_max_{window}'] = gb.shift(1).rolling(window).max()
    
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
        # 실패해도 무시 (캐시에는 추가 안 됨)

# 피처 생성
train_full_featured = create_features(train_df)

# 예측 결과를 저장할 딕셔너리 (메뉴별로 분리)
menu_predictions = {}

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
        # ✅ 수정: 메뉴별 예측 결과를 딕셔너리에 저장
        menu_predictions[menu_name] = []
        
        # 1. 메뉴별 데이터 준비 (훈련 데이터만 사용)
        menu_train_data = train_full_featured[train_df['영업장명_메뉴명'] == menu_name].copy()
        
        if len(menu_train_data) < 30:
            # 데이터가 부족한 메뉴는 0으로 예측
            for path in test_paths:
                basename = os.path.basename(path).replace('.csv', '')
                for i in range(7):
                    menu_predictions[menu_name].append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0
                    })
            continue
            
        # 2. 메뉴별 모델 학습
        predictor_path = f'autogluon_models_menu/{menu_name.replace("/", "_").replace(" ", "")}'
        predictor = None
        
        try:
            predictor = TabularPredictor.load(predictor_path)
        except:
            pass
        
        if predictor is None:
            print(f"🚀 [1단계-탐색] '{menu_name}' 메뉴의 모델 탐색을 시작합니다.")
            
            # ✅ 수정: CPU 전용 안정 설정
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
                label='매출수량',
                eval_metric='mean_absolute_error',
                path=predictor_path
            )
            
            predictor.fit(
                train_data=menu_train_data,
                hyperparameters=hyperparameters,
                time_limit=600,
                presets='medium_quality',
                num_cpus=6,
                num_gpus=0  # CPU 전용
            )
        
        # 3. 메뉴별 예측 수행
        for path in test_paths:
            basename = os.path.basename(path).replace('.csv', '')
            
            # ✅ 수정: 평가 데이터만 사용 (훈련 데이터와 분리)
            test_data = pd.read_csv(path)
            test_menu_data = test_data[test_data['영업장명_메뉴명'] == menu_name].copy()
            
            if test_menu_data.empty:
                # 해당 메뉴가 테스트에 없는 경우 0으로 예측
                for i in range(7):
                    menu_predictions[menu_name].append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0
                    })
                continue
            
            # 7일 예측 (각각 독립적으로)
            for i in range(7):
                # 예측할 날짜 계산
                last_date = pd.to_datetime(test_menu_data['영업일자'].max())
                target_date = last_date + pd.Timedelta(days=i+1)
                
                # ✅ 수정: 올바른 피처 생성 방식
                # 새로운 예측 행을 추가하여 피처 생성
                new_row = pd.DataFrame([{
                    '영업일자': target_date,
                    '영업장명_메뉴명': menu_name,
                    '매출수량': 0  # 예측할 값이므로 0으로 설정
                }])
                
                # 28일 데이터 + 새로운 행으로 피처 생성
                combined_data = pd.concat([test_menu_data, new_row], ignore_index=True)
                featured_data = create_features(combined_data)
                
                # 마지막 행의 피처만 사용하여 예측
                last_features = featured_data.tail(1).copy()
                
                # 시간 관련 피처만 업데이트
                last_features['일'] = target_date.day
                last_features['월'] = target_date.month
                last_features['요일'] = target_date.weekday()
                last_features['주'] = target_date.isocalendar()[1]
                last_features['분기'] = (target_date.month - 1) // 3 + 1
                last_features['연'] = target_date.year
                
                # 삼각함수 기반 계절성 피처 업데이트
                last_features['월_sin'] = np.sin(2 * np.pi * target_date.month / 12)
                last_features['월_cos'] = np.cos(2 * np.pi * target_date.month / 12)
                last_features['요일_sin'] = np.sin(2 * np.pi * target_date.weekday() / 7)
                last_features['요일_cos'] = np.cos(2 * np.pi * target_date.weekday() / 7)
                
                # AutoGluon 예측
                X_test = last_features.drop(['영업일자', '영업장명_메뉴명', '매출수량'], axis=1, errors='ignore')
                prediction_ag = predictor.predict(X_test).iloc[0]
                
                # Prophet 예측 (옵션)
                prediction_prophet = 0
                if Prophet is not None and len(menu_train_data) >= 50:
                    try:
                        # Prophet 모델 적합
                        prophet_data = menu_train_data[['영업일자', '매출수량']].copy()
                        prophet_data.columns = ['ds', 'y']
                        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                        
                        model = Prophet(
                            changepoint_prior_scale=0.05,
                            seasonality_prior_scale=5.0,
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                        )
                        model.fit(prophet_data)
                        
                        # Prophet 예측
                        future = pd.DataFrame({'ds': [target_date]})
                        forecast = model.predict(future)
                        prediction_prophet = forecast['yhat'].iloc[0]
                    except:
                        prediction_prophet = 0
                
                # 가중 앙상블
                if prediction_prophet > 0:
                    pred_final = PROPHET_WEIGHT * prediction_prophet + (1 - PROPHET_WEIGHT) * prediction_ag
                else:
                    pred_final = prediction_ag
                
                # 예측값이 음수인 경우 0으로 조정
                pred_final = max(0, pred_final)
                
                # ✅ 수정: 메뉴별 예측 결과를 딕셔너리에 저장
                menu_predictions[menu_name].append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred_final
                })

# ✅ 수정: 모든 메뉴의 예측 결과를 하나의 리스트로 합치기
all_predictions = []
for menu_name in unique_menus:
    all_predictions.extend(menu_predictions[menu_name])

# 예측 결과를 제출 형식으로 변환
if all_predictions:
    pred_df = pd.DataFrame(all_predictions)
    
    print("\n제출 파일 생성 중...")
    
    # ✅ 수정: 중복 제거 후 pivot
    # 중복된 예측이 있다면 마지막 값을 유지
    pred_df = pred_df.drop_duplicates(subset=['영업일자', '영업장명_메뉴명'], keep='last')
    
    submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
    
    final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
    final_submission = final_submission.fillna(0)
    
    # ✅ 수정: 안전한 컬럼 순서 매칭
    try:
        # 샘플 파일의 컬럼 순서로 맞추기
        available_columns = [col for col in submission_df.columns if col in final_submission.columns]
        final_submission = final_submission[available_columns]
        
        # 누락된 컬럼은 0으로 채우기
        missing_columns = [col for col in submission_df.columns if col not in final_submission.columns]
        for col in missing_columns:
            final_submission[col] = 0
        
        # 최종 순서 맞추기
        final_submission = final_submission[submission_df.columns]
        
    except Exception as e:
        print(f"⚠️ 컬럼 순서 매칭 실패: {e}")
        print("기본 순서로 저장합니다.")
        # 기본 순서로 저장
        pass
    
    final_submission.to_csv('submission_autogluon_menu_pipeline.csv', index=False)
    print("✅ submission_autogluon_menu_pipeline.csv 파일 생성 완료")

    # 최종 평균 검증 점수
    if all_menu_scores:
        avg_mae = np.mean(all_menu_scores)
        print(f"\n📊 전체 메뉴의 평균 검증 MAE 점수: {avg_mae:.4f}")
else:
    print("\n생성된 예측이 없습니다.")

print("\n===  AutoGluon(+Prophet 보조 앙상블) 모델 ===")
print("✅ 각 메뉴별 AutoGluon 모델 + Prophet 보조 예측 가중 앙상블")
print("✅ Prophet 미설치/데이터 부족 시 자동으로 AutoGluon 단독 예측")
print("✅ 공휴일/주간/연간 시즌성과 추세를 보조로 반영")
print("✅ Data Leakage 방지 및 PicklingError 해결")

