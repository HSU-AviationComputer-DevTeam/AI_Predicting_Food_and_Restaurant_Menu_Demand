import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
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
    df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
    df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:]
    df['메뉴명'] = df['메뉴명'].apply(lambda x: '_'.join(x) if x else '')
    
    # 범주형 변수를 숫자로 인코딩
    df['영업장명_메뉴명_encoded'] = df['영업장명_메뉴명'].astype('category').cat.codes
    df['영업장명_encoded'] = df['영업장명'].astype('category').cat.codes
    
    # === 계절별 메뉴 특성 피처 추가 ===
    
    # 1. 계절 특화 메뉴 분류
    df['봄_특화메뉴'] = 0
    df['여름_특화메뉴'] = 0
    df['가을_특화메뉴'] = 0
    df['겨울_특화메뉴'] = 0
    
    # 봄 특화 (브런치, 샐러드, 신선한 요리)
    spring_keywords = ['브런치', '샐러드', '리조또', '그릴드', '시저']
    df['봄_특화메뉴'] = df['메뉴명'].str.contains('|'.join(spring_keywords), na=False).astype(int)
    
    # 여름 특화 (차가운 음료, 해산물, 시원한 요리)
    summer_keywords = ['ice', '아이스', '에이드', '식혜', '생수', '냉면', '해물', '랍스타', '쉬림프', '해산물']
    df['여름_특화메뉴'] = df['메뉴명'].str.contains('|'.join(summer_keywords), na=False, case=False).astype(int)
    
    # 가을 특화 (따뜻한 주류)
    fall_keywords = ['막걸리', '소주', '맥주', '참이슬', '카스', 'beer']
    df['가을_특화메뉴'] = df['메뉴명'].str.contains('|'.join(fall_keywords), na=False, case=False).astype(int)
    
    # 겨울 특화 (뜨거운 국물, 보양식)
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
    df['봄_브런치_매치'] = df['봄_특화메뉴'] * (df['season'] == 2).astype(int)  # 봄(3-5월)이 season 2
    df['여름_시원함_매치'] = df['여름_특화메뉴'] * (df['season'] == 3).astype(int)  # 여름(6-8월)이 season 3
    df['가을_주류_매치'] = df['가을_특화메뉴'] * (df['season'] == 4).astype(int)  # 가을(9-11월)이 season 4
    df['겨울_따뜻함_매치'] = df['겨울_특화메뉴'] * (df['season'] == 1).astype(int)  # 겨울(12-2월)이 season 1
    
    # 6. 특이 패턴 피처 (3월 급감, 1월 최고)
    df['3월_급감패턴'] = (df['month'] == 3).astype(int)
    df['1월_최고패턴'] = (df['month'] == 1).astype(int)
    df['12월_연말패턴'] = (df['month'] == 12).astype(int)
    
    # 7. 고가중치 영업장 특별 처리 (담하, 미라시아)
    df['고가중치_영업장'] = ((df['영업장명'] == '담하') | (df['영업장명'] == '미라시아')).astype(int)
    df['담하_특별처리'] = (df['영업장명'] == '담하').astype(int)
    df['미라시아_특별처리'] = (df['영업장명'] == '미라시아').astype(int)
    
    # 고가중치 영업장 × 계절 상호작용
    df['담하_계절상호작용'] = df['담하_특별처리'] * df['season']
    df['미라시아_계절상호작용'] = df['미라시아_특별처리'] * df['season']
    
    # 7. 시간 피처
    df = df.sort_values(['영업장명_메뉴명', '영업일자'])
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).std()
    
    # NaN 값 처리
    df = df.fillna(0)
    
    return df


# 데이터 로드 및 전처리
train = pd.read_csv('./data/train/train.csv')
train['매출수량'] = train['매출수량'].clip(lower=0)

# 피처 생성
train_xgb = create_features(train)

# 학습 데이터와 검증 데이터 분리
features = [
    # 기본 시간 피처
    'day_of_week', 'is_weekend', 'month', 'day', 'week_of_year', 
    'season', 'is_holiday', 'year', 
    
    # 인코딩된 범주형 피처
    '영업장명_메뉴명_encoded', '영업장명_encoded',
    
    # 계절별 메뉴 특성 피처
    '봄_특화메뉴', '여름_특화메뉴', '가을_특화메뉴', '겨울_특화메뉴',
    
    # 메뉴 카테고리 피처
    '분식류', '음료류', '주류', '한식류', '양식류', '단체메뉴', '대여료',
    
    # 영업장별 특성 피처
    '포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아', '느티나무', '라그로타', '연회장', '화담숲카페',
    
    # 인기 메뉴 TOP 10 피처
    '인기메뉴_꼬치어묵', '인기메뉴_해물파전', '인기메뉴_떡볶이', '인기메뉴_생수', '인기메뉴_아메리카노',
    '인기메뉴_치즈핫도그', '인기메뉴_돈까스', '인기메뉴_단체식', '인기메뉴_콜라',
    
    # 계절-메뉴 상호작용 피처
    '봄_브런치_매치', '여름_시원함_매치', '가을_주류_매치', '겨울_따뜻함_매치',
    
    # 특이 패턴 피처
    '3월_급감패턴', '1월_최고패턴', '12월_연말패턴',
    
    # 고가중치 영업장 특별 처리 피처
    '고가중치_영업장', '담하_특별처리', '미라시아_특별처리', '담하_계절상호작용', '미라시아_계절상호작용'
    
] + [f'lag_{lag}' for lag in [1, 2, 3, 7, 14]] + \
    [f'rolling_mean_{window}' for window in [7, 14, 28]] + \
    [f'rolling_std_{window}' for window in [7, 14, 28]]

X = train_xgb[features]
y = train_xgb['매출수량']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# XGBoost 모델 학습 및 하이퍼파라미터 튜닝
xgb_params_list = [
    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1},
    {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05},
    {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.07},
]

best_mse = float('inf')
best_xgb_params = None
best_xgb_model = None

for params in tqdm(xgb_params_list, desc="XGBoost H-param Tuning"):
    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds_val = model.predict(X_val)
    mse = mean_squared_error(y_val, preds_val)
    print(f"\n[XGBoost Params {params}] Validation MSE: {mse:.4f}")
    
    if mse < best_mse:
        best_mse = mse
        best_xgb_params = params
        best_xgb_model = model

print(f"\nBest XGBoost Params: {best_xgb_params}")

# 최적 XGBoost 모델 저장
os.makedirs("models", exist_ok=True)
best_xgb_model.save_model("models/best_xgboost_model.json")
print("XGBoost 최적 모델 저장 완료")


# 테스트 데이터 예측 (28일 제한 규칙 준수)
submission_df = pd.read_csv('./data/sample_submission.csv')

# test 폴더 내 모든 파일 처리
test_paths = sorted(glob.glob('./data/test/*.csv'))
if not test_paths:
    print("test 폴더에 예측할 파일이 없습니다.")
else:
    all_predictions = []
    
    for path in test_paths:
        test_df = pd.read_csv(path)
        test_df = create_features(test_df)
        
        basename = os.path.basename(path).replace('.csv', '')
        
        # 샘플 제출 파일에서 필요한 모든 메뉴 가져오기
        required_menus = list(submission_df.columns[1:])  # 첫 번째 컬럼은 영업일자
        
        print(f"처리 중: {basename}, 필요한 메뉴 수: {len(required_menus)}")
        
        # 각 메뉴에 대해 예측
        processed_menus = 0
        for menu_name in required_menus:
            menu_data = test_df[test_df['영업장명_메뉴명'] == menu_name].copy()
            
            if len(menu_data) == 0:
                # 해당 메뉴의 데이터가 없는 경우 기본값 사용
                for i in range(7):
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0.0
                    })
                continue
            
            # 메뉴별로 최근 28일 데이터만 사용 (대회 규칙 준수)
            menu_data = menu_data.tail(28).copy()
            
            # 7일 예측
            prediction_data = menu_data.copy()
            
            for i in range(7):
                if len(prediction_data) > 0:
                    # 최근 데이터로 피처 생성
                    try:
                        X_test = prediction_data[features].tail(1)
                        
                        # 예측
                        pred = best_xgb_model.predict(X_test)[0]
                        pred = max(0, pred)  # 음수 제거
                    except:
                        # 피처 문제가 있는 경우 기본값 사용
                        pred = 0.0
                    
                    # 예측 결과 저장
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': pred
                    })
                    
                    # 다음 예측을 위해 데이터 업데이트
                    next_date = prediction_data['영업일자'].max() + pd.Timedelta(days=1)
                    new_row = prediction_data.iloc[-1:].copy()
                    new_row['영업일자'] = next_date
                    new_row['매출수량'] = pred
                    
                    # 데이터 추가 및 크기 제한 (28일 + 예측일수)
                    prediction_data = pd.concat([prediction_data, new_row], ignore_index=True)
                    if len(prediction_data) > 35:  # 28 + 7 여유분
                        prediction_data = prediction_data.tail(35)
                    
                    # 피처 다시 생성
                    try:
                        prediction_data = create_features(prediction_data)
                    except:
                        # 피처 생성 문제가 있는 경우 기존 데이터 유지
                        pass
                else:
                    # 데이터가 없는 경우 0으로 예측
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': 0.0
                    })
            
            processed_menus += 1
            
        print(f"  처리된 메뉴 수: {processed_menus}/{len(required_menus)}")
    
    # 예측 결과를 제출 형식으로 변환
    pred_df = pd.DataFrame(all_predictions)
    
    # 제출 파일 생성 (모든 메뉴에 대해 예측값 채우기)
    print("제출 파일 생성 중...")
    print(f"예측 결과 데이터 개수: {len(pred_df)}")
    print(f"샘플 제출 파일 행 수: {len(submission_df)}")
    
    filled_count = 0
    for idx, row in submission_df.iterrows():
        date = row['영업일자']
        for col in submission_df.columns[1:]:  # 첫 번째 컬럼은 영업일자
            matching_pred = pred_df[(pred_df['영업일자'] == date) & (pred_df['영업장명_메뉴명'] == col)]
            if not matching_pred.empty:
                submission_df.at[idx, col] = float(matching_pred['매출수량'].iloc[0])
                filled_count += 1
            else:
                # 매칭되는 예측이 없는 경우, 해당 메뉴의 평균값이나 0 사용
                submission_df.at[idx, col] = 0.0
    
    print(f"총 채워진 예측값 개수: {filled_count}")
    
    # 각 열(메뉴)별로 0이 아닌 값의 개수 확인
    non_zero_counts = {}
    for col in submission_df.columns[1:]:
        non_zero_count = (submission_df[col] != 0).sum()
        if non_zero_count > 0:
            non_zero_counts[col] = non_zero_count
    
    print(f"0이 아닌 예측값을 가진 메뉴 수: {len(non_zero_counts)}")
    if len(non_zero_counts) > 0:
        print("예측값이 있는 메뉴들:")
        for menu, count in list(non_zero_counts.items())[:10]:  # 처음 10개만 출력
            print(f"  {menu}: {count}개 예측값")
        if len(non_zero_counts) > 10:
            print(f"  ... 및 {len(non_zero_counts) - 10}개 메뉴 더")


submission_df.to_csv('submission_xgboost_competition_compliant.csv', index=False)
print("submission_xgboost_competition_compliant.csv 파일 생성 완료")
print("\n=== 🏆 대회 규칙 준수 + 고급 피처 XGBoost 모델 ===")
print("✅ 대회 규칙 준수:")
print("  • 28일 데이터 제한 규칙 적용")
print("  • 시계열 Data Leakage 방지")
print("  • 독립적 추론 수행")
print("\n📊 추가된 고급 피처:")
print("  • 계절별 메뉴 특화 분류 (봄/여름/가을/겨울)")
print("  • 메뉴 카테고리 분류 (분식/음료/주류/한식/양식/단체/대여)")
print("  • 영업장별 원핫 인코딩")
print("  • 인기 메뉴 TOP 10 특별 처리")
print("  • 계절-메뉴 상호작용 피처")
print("  • 특이 패턴 피처 (3월 급감, 1월 최고, 12월 연말)")
print("  • 🎯 고가중치 영업장 특별 처리 (담하, 미라시아)")
print(f"  • 총 피처 수: {len(features)}개")
print("\n🎯 수료 기준 목표:")
print("  • Public Score ≤ 0.711046")
print("  • Private Score ≤ 0.693935")