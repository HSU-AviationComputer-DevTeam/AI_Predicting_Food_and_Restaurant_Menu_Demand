import os
import glob
import random
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


# 피처 생성 함수 (계절별 메뉴 특성 추가)
def create_features(df):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
    df['week_of_year'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)
    df['year'] = df['영업일자'].dt.year

    # 영업장명과 메뉴명 분리
    if '영업장명_메뉴명' in df.columns:
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].apply(lambda x: '_'.join(x) if x else '')

        # 범주형 변수를 AutoGluon이 처리하도록 문자열로 유지
        # df['영업장명_메뉴명_encoded'] = df['영업장명_메뉴명'].astype('category').cat.codes
        # df['영업장명_encoded'] = df['영업장명'].astype('category').cat.codes

    # === 계절별 메뉴 특성 피처 추가 ===
    if '메뉴명' in df.columns:
        # 1. 계절 특화 메뉴 분류
        df['봄_특화메뉴'] = 0
        df['여름_특화메뉴'] = 0
        df['가을_특화메뉴'] = 0
        df['겨울_특화메뉴'] = 0
        
        spring_keywords = ['브런치', '샐러드', '리조또', '그릴드', '시저']
        df['봄_특화메뉴'] = df['메뉴명'].str.contains('|'.join(spring_keywords), na=False).astype(int)
        
        summer_keywords = ['ice', '아이스', '에이드', '식혜', '생수', '냉면', '해물', '랍스타', '쉬림프', '해산물']
        df['여름_특화메뉴'] = df['메뉴명'].str.contains('|'.join(summer_keywords), na=False, case=False).astype(int)
        
        fall_keywords = ['막걸리', '소주', '맥주', '참이슬', '카스', 'beer']
        df['가을_특화메뉴'] = df['메뉴명'].str.contains('|'.join(fall_keywords), na=False, case=False).astype(int)
        
        winter_keywords = ['국', '탕', '찌개', '해장', 'hot', '핫도그', '떡볶이', '꼬치어묵', '파전', '불고기', '갈비', '돈까스', 'bbq', '한우', '삼겹']
        df['겨울_특화메뉴'] = df['메뉴명'].str.contains('|'.join(winter_keywords), na=False, case=False).astype(int)
        
        # 2. 메뉴 카테고리 분류
        df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False).astype(int)
        df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료', na=False).astype(int)
        df['주류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|beer|생맥주', na=False, case=False).astype(int)
        df['한식류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥', na=False).astype(int)
        df['양식류'] = df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또|스파게티', na=False).astype(int)
        df['단체메뉴'] = df['메뉴명'].str.contains('단체|패키지|세트|브런치', na=False).astype(int)
        df['대여료'] = df['메뉴명'].str.contains('대여료|이용료|conference|convention', na=False, case=False).astype(int)
        
        # 3. 영업장별 특성
        df['포레스트릿'] = (df['영업장명'] == '포레스트릿').astype(int)
        df['카페테리아'] = (df['영업장명'] == '카페테리아').astype(int)
        df['화담숲주막'] = (df['영업장명'] == '화담숲주막').astype(int)
        df['담하'] = (df['영업장명'] == '담하').astype(int)
        df['미라시아'] = (df['영업장명'] == '미라시아').astype(int)
        df['느티나무'] = (df['영업장명'] == '느티나무 셀프BBQ').astype(int)
        df['라그로타'] = (df['영업장명'] == '라그로타').astype(int)
        df['연회장'] = (df['영업장명'] == '연회장').astype(int)
        df['화담숲카페'] = (df['영업장명'] == '화담숲카페').astype(int)
        
        # 4. 인기 메뉴 TOP 10 특별 처리
        df['인기메뉴_꼬치어묵'] = df['메뉴명'].str.contains('꼬치어묵', na=False).astype(int)
        df['인기메뉴_해물파전'] = df['메뉴명'].str.contains('해물파전', na=False).astype(int)
        df['인기메뉴_떡볶이'] = df['메뉴명'].str.contains('떡볶이', na=False).astype(int)
        df['인기메뉴_생수'] = df['메뉴명'].str.contains('생수', na=False).astype(int)
        df['인기메뉴_아메리카노'] = df['메뉴명'].str.contains('아메리카노', na=False).astype(int)
        df['인기메뉴_치즈핫도그'] = df['메뉴명'].str.contains('치즈 핫도그', na=False).astype(int)
        df['인기메뉴_돈까스'] = df['메뉴명'].str.contains('돈까스', na=False).astype(int)
        df['인기메뉴_단체식'] = df['메뉴명'].str.contains('단체식', na=False).astype(int)
        df['인기메뉴_콜라'] = df['메뉴명'].str.contains('콜라', na=False).astype(int)
        
        # 5. 계절-메뉴 상호작용 피처
        df['봄_브런치_매치'] = df['봄_특화메뉴'] * (df['season'] == 2).astype(int)
        df['여름_시원함_매치'] = df['여름_특화메뉴'] * (df['season'] == 3).astype(int)
        df['가을_주류_매치'] = df['가을_특화메뉴'] * (df['season'] == 4).astype(int)
        df['겨울_따뜻함_매치'] = df['겨울_특화메뉴'] * (df['season'] == 1).astype(int)
        
        # 6. 특이 패턴 피처
        df['3월_급감패턴'] = (df['month'] == 3).astype(int)
        df['1월_최고패턴'] = (df['month'] == 1).astype(int)
        df['12월_연말패턴'] = (df['month'] == 12).astype(int)
        
        # 7. 고가중치 영업장 특별 처리
        df['고가중치_영업장'] = ((df['영업장명'] == '담하') | (df['영업장명'] == '미라시아')).astype(int)
        df['담하_특별처리'] = (df['영업장명'] == '담하').astype(int)
        df['미라시아_특별처리'] = (df['영업장명'] == '미라시아').astype(int)
        
        # 고가중치 영업장 × 계절 상호작용
        df['담하_계절상호작용'] = df['담하_특별처리'] * df['season']
        df['미라시아_계절상호작용'] = df['미라시아_특별처리'] * df['season']

    # 시간 피처
    sort_keys = ['영업장명_메뉴명', '영업일자'] if '영업장명_메뉴명' in df.columns else ['영업일자']
    df = df.sort_values(sort_keys)
    
    if '매출수량' in df.columns:
        gb_key = '영업장명_메뉴명' if '영업장명_메뉴명' in df.columns else '영업일자'
        gb = df.groupby(gb_key)['매출수량']
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = gb.shift(lag)
        for window in [7, 14, 28]:
            df[f'rolling_mean_{window}'] = gb.shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = gb.shift(1).rolling(window).std()
    
    # AutoGluon이 날짜 관련 피처를 범주형으로 인식하도록 타입 변경
    df['month'] = df['month'].astype(str)
    df['day'] = df['day'].astype(str)
    df['week_of_year'] = df['week_of_year'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['year'] = df['year'].astype(str)
    df['season'] = df['season'].astype(str)

    # 불필요한 원본 컬럼 제거
    if '메뉴명' in df.columns:
        df = df.drop(columns=['영업장명', '메뉴명'])

    df = df.fillna(0)
    
    return df

# 데이터 로드
train_df = pd.read_csv('./data/train/train.csv')
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
submission_df = pd.read_csv('./data/sample_submission.csv')

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
    
    for menu_name in tqdm(unique_menus, desc="메뉴별 모델 학습 및 예측"):
        
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
                eval_metric='root_mean_squared_error'
            ).fit(
                train_data_ag,
                presets='best_quality',
                time_limit=180, # 메뉴별 최대 학습 시간(초)
                ag_args_fit={'num_gpus': 0}
            )

        # 3. 메뉴별 테스트 데이터 예측 (순환 예측)
        for path in test_paths:
            test_file_df = pd.read_csv(path)
            
            if menu_name not in test_file_df['영업장명_메뉴명'].unique():
                continue

            basename = os.path.basename(path).replace('.csv', '')
            
            # 예측에 사용할 과거 데이터 (train + test 파일의 과거 데이터)
            historical_data = pd.concat([
                train_full_featured[train_full_featured['영업장명_메뉴명'] == menu_name],
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
                update_row = featured_data.tail(1).copy()
                update_row['매출수량'] = pred
                historical_data = pd.concat([historical_data, update_row], ignore_index=True)

# 예측 결과를 제출 형식으로 변환
if all_predictions:
    pred_df = pd.DataFrame(all_predictions)
    
    print("제출 파일 생성 중...")
    
    submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
    
    final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
    final_submission = final_submission.fillna(0)
    
    # 컬럼 순서를 샘플과 동일하게 맞춤
    final_submission = final_submission[submission_df.columns]
    
    final_submission.to_csv('submission_autogluon_per_item.csv', index=False)
    print("submission_autogluon_per_item.csv 파일 생성 완료")
else:
    print("생성된 예측이 없습니다.")

print("\n=== 🏆 AutoGluon (메뉴별 모델)을 이용한 자동화 모델 ===")
print("✅ 각 메뉴별로 독립적인 모델을 학습하여 예측 정확도 향상 시도")
print("✅ AutoGluon이 자동으로 최적의 모델 탐색 및 앙상블 수행")
print("\n🎯 수료 기준 목표:")
print("  • Public Score ≤ 0.711046")
print("  • Private Score ≤ 0.693935")
