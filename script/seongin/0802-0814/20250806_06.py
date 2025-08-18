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


# 피처 생성 함수
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

    # 범주형 변수를 숫자로 인코딩
    df['영업장명_메뉴명_encoded'] = df['영업장명_메뉴명'].astype('category').cat.codes
    
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
    'day_of_week', 'is_weekend', 'month', 'day', 'week_of_year', 
    'season', 'is_holiday', 'year', '영업장명_메뉴명_encoded'
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


# 테스트 데이터 예측
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
        
        # 각 메뉴에 대해 예측
        for menu_name in test_df['영업장명_메뉴명'].unique():
            menu_data = test_df[test_df['영업장명_메뉴명'] == menu_name].copy()
            
            # 7일 예측
            for i in range(7):
                if len(menu_data) > 0:
                    # 최근 데이터로 피처 생성
                    X_test = menu_data[features].tail(1)
                    
                    # 예측
                    pred = best_xgb_model.predict(X_test)[0]
                    pred = max(0, pred)  # 음수 제거
                    
                    # 예측 결과 저장
                    all_predictions.append({
                        '영업일자': f"{basename}+{i+1}일",
                        '영업장명_메뉴명': menu_name,
                        '매출수량': pred
                    })
                    
                    # 다음 예측을 위해 데이터 업데이트
                    next_date = menu_data['영업일자'].max() + pd.Timedelta(days=1)
                    new_row = menu_data.iloc[-1:].copy()
                    new_row['영업일자'] = next_date
                    new_row['매출수량'] = pred
                    
                    # 피처 다시 생성
                    new_row = create_features(pd.concat([menu_data, new_row], ignore_index=True)).tail(1)
                    menu_data = pd.concat([menu_data, new_row], ignore_index=True)
    
    # 예측 결과를 제출 형식으로 변환
    pred_df = pd.DataFrame(all_predictions)
    
    # 제출 파일 생성
    for idx, row in submission_df.iterrows():
        date = row['영업일자']
        for col in submission_df.columns[1:]:  # 첫 번째 컬럼은 영업일자
            matching_pred = pred_df[(pred_df['영업일자'] == date) & (pred_df['영업장명_메뉴명'] == col)]
            if not matching_pred.empty:
                submission_df.at[idx, col] = float(matching_pred['매출수량'].iloc[0])


submission_df.to_csv('submission_xgboost.csv', index=False)
print("submission_xgboost.csv 파일 생성 완료")