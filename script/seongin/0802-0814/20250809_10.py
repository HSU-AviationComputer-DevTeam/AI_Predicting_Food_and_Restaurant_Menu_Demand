import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


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


# 피처 생성 함수 (TimeSeriesPredictor에 맞게 수정)
def create_features(df, is_train=True):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    
    # 시간 관련 피처 (known covariates)
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
    df['week_of_year'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)
    df['year'] = df['영업일자'].dt.year

    # 정적 피처 (static features) - is_train일 때만 생성
    if is_train and '영업장명_메뉴명' in df.columns:
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].apply(lambda x: '_'.join(x) if x else '')

        # 메뉴 카테고리 분류
        df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False).astype(int)
        df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료', na=False).astype(int)
        df['주류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|beer|생맥주', na=False, case=False).astype(int)
        df['한식류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥', na=False).astype(int)
        df['양식류'] = df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또|스파게티', na=False).astype(int)
        df['단체메뉴'] = df['메뉴명'].str.contains('단체|패키지|세트|브런치', na=False).astype(int)
        df['대여료'] = df['메뉴명'].str.contains('대여료|이용료|conference|convention', na=False, case=False).astype(int)
        
        # 영업장별 특성
        df['영업장_카테고리'] = df['영업장명'].astype('category').cat.codes

        df = df.drop(columns=['영업장명', '메뉴명'])

    # TimeSeriesPredictor는 내부적으로 lag, rolling 피처를 생성하므로 수동 생성 부분 제거
    
    # AutoGluon이 날짜 관련 피처를 범주형으로 인식하도록 타입 변경
    df['month'] = df['month'].astype(str)
    df['day'] = df['day'].astype(str)
    df['week_of_year'] = df['week_of_year'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['year'] = df['year'].astype(str)
    df['season'] = df['season'].astype(str)
    
    return df

# 데이터 로드
print("데이터 로딩 중...")
train_df = pd.read_csv('./data/train/train.csv')
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
submission_df = pd.read_csv('./data/sample_submission.csv')

# 피처 생성
print("피처 생성 중...")
train_featured = create_features(train_df, is_train=True)

# TimeSeriesDataFrame으로 변환
ts_df = TimeSeriesDataFrame.from_data_frame(
    train_featured,
    id_column='영업장명_메뉴명',
    timestamp_column='영업일자'
)

# TimeSeriesPredictor 학습
predictor_path = 'autogluon_timeseries_model'
predictor = TimeSeriesPredictor(
    label='매출수량',
    path=predictor_path,
    prediction_length=7, # 7일 예측
    eval_metric='RMSE',
    known_covariates_names=[
        'day_of_week', 'is_weekend', 'month', 'day', 
        'week_of_year', 'season', 'is_holiday', 'year'
    ]
)

print("🚀 새로운 통합 모델 학습 시작...")
predictor.fit(
    ts_df,
    presets='best_quality',
    time_limit=600 # 학습 시간 600초로 증가
)

# 예측
print("예측 생성 중...")
all_predictions_list = []

test_paths = sorted(glob.glob('./data/test/*.csv'))
if not test_paths:
    print("test 폴더에 예측할 파일이 없습니다.")
else:
    for path in tqdm(test_paths, desc="Test 파일별 예측 생성"):
        test_file_df = pd.read_csv(path)
        basename = os.path.basename(path).replace('.csv', '')
        
        # 1. 예측을 위한 컨텍스트 데이터 준비 (train + test)
        menus_in_test = test_file_df['영업장명_메뉴명'].unique()
        
        # train 데이터에서 현재 test 파일에 존재하는 메뉴들의 데이터만 필터링
        historical_train_data = train_featured[train_featured['영업장명_메뉴명'].isin(menus_in_test)]
        
        # feature가 생성된 test 데이터 준비
        # TimeSeriesPredictor는 feature를 내부적으로 처리하므로 원본 test_file_df 사용
        context_df = pd.concat([historical_train_data, test_file_df], ignore_index=True)

        context_ts_df = TimeSeriesDataFrame.from_data_frame(
            context_df,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자"
        )

        # 2. 미래의 known_covariates 생성
        last_date = pd.to_datetime(test_file_df['영업일자']).max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
        
        # 모든 메뉴에 대해 미래 날짜 데이터프레임 생성
        future_df_list = []
        for menu in menus_in_test:
            temp_df = pd.DataFrame({'영업장명_메뉴명': [menu] * 7, '영업일자': future_dates})
            future_df_list.append(temp_df)
        
        future_df = pd.concat(future_df_list, ignore_index=True)
        future_known_covariates_df = create_features(future_df, is_train=False)

        # TimeSeriesDataFrame으로 변환하여 id_column 명시
        future_known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_known_covariates_df,
            id_column="영업장명_메뉴명",
            timestamp_column="영업일자",
        )

        # 3. 예측 실행
        predictions = predictor.predict(
            context_ts_df,
            known_covariates=future_known_covariates
        )
        
        # 4. 예측 결과 포맷팅
        predictions['mean'] = predictions['mean'].clip(lower=0)
        pred_df_single_test = predictions.reset_index()

        pred_df_single_test['day_offset'] = pred_df_single_test.groupby('item_id').cumcount()
        pred_df_single_test['영업일자'] = pred_df_single_test.apply(
            lambda row: f"{basename}+{row['day_offset']+1}일", axis=1
        )
        pred_df_single_test = pred_df_single_test.rename(columns={'item_id': '영업장명_메뉴명', 'mean': '매출수량'})
        
        all_predictions_list.append(pred_df_single_test[['영업일자', '영업장명_메뉴명', '매출수량']])

# 최종 제출 파일 생성
if all_predictions_list:
    print("제출 파일 생성 중...")
    
    # 모든 예측 결과 결합
    final_pred_df = pd.concat(all_predictions_list, ignore_index=True)

    # Pivot 테이블 생성
    submission_pivot = final_pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()

    # 최종 제출 파일 생성
    final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
    final_submission = final_submission.fillna(0)
    
    # 컬럼 순서를 샘플과 동일하게 맞춤
    final_submission = final_submission[submission_df.columns]
    
    output_filename = 'submission_autogluon_timeseries.csv'
    final_submission.to_csv(output_filename, index=False)
    print(f"{output_filename} 파일 생성 완료")

print("\n=== 🏆 AutoGluon-TimeSeries (통합 모델)을 이용한 자동화 모델 ===")
print("✅ 모든 메뉴를 하나의 시계열 모델로 학습하여 예측 정확도 향상 시도")
print("✅ AutoGluon이 자동으로 최적의 시계열 모델 탐색 및 앙상블 수행")
print("✅ 학습 시간 600초로 증가, TimeSeriesPredictor 활용")
print("\n🎯 수료 기준 목표:")
print("  • Public Score ≤ 0.711046")
print("  • Private Score ≤ 0.693935")
