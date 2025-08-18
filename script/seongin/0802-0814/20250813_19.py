import os
import glob
import random
import shutil # 폴더 삭제를 위해 추가
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# --- 기존 모델 폴더 삭제 ---
# 피처 로직이 변경되었으므로, 이전 버전과 충돌을 막기 위해 기존 모델을 삭제하고 새로 학습합니다.
# if os.path.exists('autogluon_models'):
#     print("피처 로직 변경으로 인해 기존 모델 폴더('autogluon_models')를 삭제합니다.")
#     shutil.rmtree('autogluon_models')


# 공휴일 리스트 및 함수
holiday_dates = pd.to_datetime([
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
    "2025-10-07","2025-10-08","2025-10-09","2025-12-25"
])

def is_korean_holiday(date):
    return int(date in holiday_dates)


# --- 메뉴 메타데이터 (전역 변수) ---
menu_meta = {}
discontinued_menus = set()


# 피처 생성 함수 (개선)
def create_features(df):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    
    # 1. 기본 시간 피처
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
    df['week_of_year'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['year'] = df['영업일자'].dt.year
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)

    # 2. 영업장/메뉴명 관련 피처
    if '영업장명_메뉴명' in df.columns:
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].apply(lambda x: '_'.join(x) if x else '')
        
        # 메뉴 카테고리 분류 (핵심 키워드 기반)
        df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False).astype(int)
        df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료|식혜', na=False, case=False).astype(int)
        df['주류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|beer|생맥주', na=False, case=False).astype(int)
        df['식사류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥|파스타|피자|스테이크|샐러드|리조또', na=False).astype(int)
        df['단체/대여'] = df['메뉴명'].str.contains('단체|패키지|세트|대여료|conference|convention', na=False, case=False).astype(int)
    
    # 3. 메뉴별 특성 피처 (메타데이터 활용)
    if '영업장명_메뉴명' in df.columns and menu_meta:
        # 출시 후 경과일 & 활성 시즌 여부
        df['days_since_launch'] = -1
        df['is_active_month'] = 0
        
        for name, meta in menu_meta.items():
            menu_mask = (df['영업장명_메뉴명'] == name)
            if menu_mask.any():
                df.loc[menu_mask, 'days_since_launch'] = (df.loc[menu_mask, '영업일자'] - meta['launch_date']).dt.days
                df.loc[menu_mask, 'is_active_month'] = df.loc[menu_mask, 'month'].isin(meta['active_months']).astype(int)

    # 4. 시계열 피처 (Lag & Rolling)
    sort_keys = ['영업장명_메뉴명', '영업일자'] if '영업장명_메뉴명' in df.columns else ['영업일자']
    df = df.sort_values(sort_keys)
    
    if '매출수량' in df.columns:
        gb_key = '영업장명_메뉴명' if '영업장명_메뉴명' in df.columns else '영업일자'
        grouped = df.groupby(gb_key)['매출수량']
        for lag in [1, 7, 14]:
            df[f'lag_{lag}'] = grouped.shift(lag)
        for window in [7, 14]:
            df[f'rolling_mean_{window}'] = grouped.shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = grouped.shift(1).rolling(window).std()
    
    # 5. 후처리
    # AutoGluon이 날짜 관련 피처를 범주형으로 인식하도록 타입 변경
    df['month'] = df['month'].astype(str)
    df['day'] = df['day'].astype(str)
    df['week_of_year'] = df['week_of_year'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['year'] = df['year'].astype(str)
    
    if '메뉴명' in df.columns:
        df = df.drop(columns=['영업장명', '메뉴명'])

    df = df.fillna(0)
    
    return df

# 데이터 로드
train_df = pd.read_csv('./data/train/train.csv')
train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
submission_df = pd.read_csv('./data/sample_submission.csv')

# --- 메뉴 메타데이터 생성 ---
for name, group in train_df.groupby('영업장명_메뉴명'):
    sales_data = group[group['매출수량'] > 0]
    if not sales_data.empty:
        menu_meta[name] = {
            'launch_date': sales_data['영업일자'].min(),
            'last_sale_date': sales_data['영업일자'].max(),
            'active_months': list(sales_data['영업일자'].dt.month.unique())
        }

# 단종 메뉴 식별 (학습 데이터 마지막 날짜 기준 60일 이상 판매 기록 없는 메뉴)
last_train_date = train_df['영업일자'].max()
discontinued_menus = {
    name for name, meta in menu_meta.items()
    if (last_train_date - meta['last_sale_date']).days > 60
}
print(f"총 {len(discontinued_menus)}개의 단종 추정 메뉴 식별.")

# 피처 생성
train_full_featured = create_features(train_df)

# 예측 결과를 저장할 리스트
all_predictions = []

# test 폴더 내 모든 파일 처리
test_paths = sorted(glob.glob('./data/test/*.csv'))
if not test_paths:
    print("test 폴더에 예측할 파일이 없습니다.")
else:
    # 각 메뉴별로 모델 학습 및 예측
    unique_menus = train_full_featured['영업장명_메뉴명'].unique()
    training_logs = [] # 학습 로그를 저장할 리스트
    
    for menu_name in tqdm(unique_menus, desc="메뉴별 모델 학습 및 예측"):
        
        # 단종 메뉴는 건너뛰기
        if menu_name in discontinued_menus:
            print(f"🗑️ 단종 메뉴로 추정되어 건너뜁니다: {menu_name}")
            continue

        # 학습 데이터에 없는 신규 메뉴 처리 (예측 단계에서 0으로 처리)
        if menu_name not in menu_meta:
            print(f"🆕 학습 데이터에 없는 신규 메뉴입니다: {menu_name}")
            continue
            
        # 1. 메뉴별 데이터 준비
        menu_train_data = train_full_featured[train_full_featured['영업장명_메뉴명'] == menu_name].copy()
        
        if len(menu_train_data) < 30: # 학습에 필요한 최소 데이터 수
            continue
            
        # AutoGluon 학습을 위한 데이터 준비
        train_data_ag = menu_train_data.drop(columns=['영업일자', '영업장명_메뉴명'])

        predictor_path = f'autogluon_models/{menu_name.replace("/", "_").replace(" ", "")}'
        # "스마트한 이어하기" 기능: 이미 학습된 모델이 있으면 불러오고, 없으면 새로 학습
        if os.path.exists(predictor_path):
            print(f"✅ 이미 학습된 모델 발견: {menu_name}. 불러오기를 시작합니다.")
            predictor = TabularPredictor.load(predictor_path)
        else:
            print(f"🚀 새로운 모델 학습 시작: {menu_name}")
            predictor = TabularPredictor(
                label='매출수량',
                path=predictor_path,
                problem_type='regression',
                eval_metric='symmetric_mean_absolute_percentage_error'
            ).fit(
                train_data_ag,
                time_limit=60, # 메뉴별 최대 학습 시간(초)
                hyperparameters={
                    'CAT': {},  # CatBoost
                    'GBM': {},  # LightGBM (LGB가 아닌 GBM이 정확한 키입니다)
                    'XGB': {},  # XGBoost
                },
                ag_args_fit={'num_gpus': 0}
            )
        
        # 모델 성능 리더보드 출력 (SMAPE 기준)
        print(f"--- {menu_name} 모델 성능 리더보드 ---")
        leaderboard = predictor.leaderboard(silent=True)
        print(leaderboard[['model', 'score_val', 'fit_time']])
        print("-------------------------------------\n")
        
        # 학습 결과 로깅
        if not leaderboard.empty:
            best_model_info = leaderboard.iloc[0]
            fit_summary = predictor.fit_summary(verbosity=0)
            training_logs.append({
                'menu_name': menu_name,
                'training_samples': len(train_data_ag),
                'best_model': best_model_info['model'],
                'validation_smape': best_model_info['score_val'],
                'total_fit_time': fit_summary.get('total_time', 0) # .get()을 사용하여 키가 없을 때 오류 방지
            })

        # 3. 메뉴별 테스트 데이터 예측 (순환 예측)
        for path in test_paths:
            test_file_df = pd.read_csv(path)
            basename = os.path.basename(path).replace('.csv', '')

            # [수정] 순환 예측 시 과거 데이터 구성 방식을 수정하여 오류를 바로잡습니다.
            # - 원인: 피처가 생성된 데이터(train_full_featured)와 원본 데이터(test_file_df)를 합쳐 컬럼 불일치 발생
            # - 해결: 피처 생성 전의 원본 train_df를 사용하여 데이터를 구성합니다.
            # - 또한, test 파일에 판매 기록이 없는 메뉴도 예측해야 하므로 관련 continue 로직을 제거합니다.
            historical_data = pd.concat([
                train_df[train_df['영업장명_메뉴명'] == menu_name], # 원본 train_df 사용
                test_file_df[test_file_df['영업장명_메뉴명'] == menu_name]
            ]).copy()
            
            historical_data['영업일자'] = pd.to_datetime(historical_data['영업일자'])
            historical_data = historical_data.sort_values(by='영업일자').tail(28) # 대회 규칙: 최근 28일 데이터 사용

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
                X_test = featured_data.tail(1).drop(columns=['영업일자', '영업장명_메뉴명', '매출수량'])
                
                # 예측
                pred = predictor.predict(X_test).iloc[0]
                pred = max(0, pred)
                
                # 예측 결과 저장
                all_predictions.append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred
                })

                # 다음 예측을 위해 예측값을 포함하여 historical_data 업데이트
                # update_row = featured_data.tail(1).copy() -> 이 방식은 피처가 포함된 데이터를 합치므로 오류 유발
                # update_row['매출수량'] = pred
                # historical_data = pd.concat([historical_data, update_row], ignore_index=True)
                
                # [수정된 로직] 핵심 정보만 포함된 행을 만들어서 historical_data에 추가
                new_prediction_row = new_row.copy()
                new_prediction_row['매출수량'] = pred
                historical_data = pd.concat([historical_data, new_prediction_row], ignore_index=True)

# 학습 로그 저장
if training_logs:
    log_df = pd.DataFrame(training_logs)
    log_df.to_csv('training_log.csv', index=False, encoding='utf-8-sig')
    print("✅ 학습 과정에 대한 training_log.csv 파일 생성 완료")

# 예측 결과를 제출 형식으로 변환
if submission_df is not None:
    # 모든 예측 결과를 하나의 데이터프레임으로 만듦
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)

        # 1. 예측 결과를 피벗 테이블로 변환
        submission_pivot = pred_df.pivot(
            index='영업일자',
            columns='영업장명_메뉴명',
            values='매출수량'
        )

        # 2. sample_submission을 최종 제출본으로 복사하고 인덱스 설정
        final_submission = submission_df.copy()
        final_submission = final_submission.set_index('영업일자')

        # 3. update 메서드를 사용하여 예측값 업데이트
        final_submission.update(submission_pivot)

        # 4. 인덱스 리셋
        final_submission.reset_index(inplace=True)

    else:
        # 예측이 없는 경우 sample_submission을 그대로 사용
        final_submission = submission_df.copy()

    # 최종적으로 모든 NaN 값을 0으로 채움 (안전장치)
    final_submission = final_submission.fillna(0)
    
    final_submission.to_csv('submission_autogluon_per_item.csv', index=False, encoding='utf-8-sig')
    print("✅ submission_autogluon_per_item.csv 파일 생성 완료")
else:
    print("생성된 예측이 없습니다.")

print("\n=== 🏆 AutoGluon (메뉴별 모델)을 이용한 자동화 모델 ===")
print("✅ 각 메뉴별로 독립적인 모델을 학습하여 예측 정확도 향상 시도")
print("✅ AutoGluon이 자동으로 최적의 모델 탐색 및 앙상블 수행")
print("✅ 데이터 기반의 메뉴 특성(출시일, 판매시즌, 단종)을 피처로 활용")
print("\n🎯 수료 기준 목표:")
print("  • Public Score ≤ 0.711046")
print("  • Private Score ≤ 0.693935")
