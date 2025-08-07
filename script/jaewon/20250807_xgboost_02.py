import os
import random
import glob
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# --- 시드 고정 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

LOOKBACK, PREDICT = 28, 7

# --- 공휴일 목록 ---
holidays = pd.to_datetime([
    # 2023, 2024 공휴일
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03", 
    "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", 
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15", 
    "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", 
    "2024-10-03", "2024-10-09", "2024-12-25"
])

# --- 날짜 파생변수 추가 ---
def add_date_features(df):
    df['영업일자_dt'] = pd.to_datetime(df['영업일자'].str.replace(r'TEST_\d+\+', '', regex=True), errors='coerce')
    df['요일'] = df['영업일자_dt'].dt.dayofweek
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['공휴일여부'] = df['영업일자_dt'].isin(holidays).astype(int)
    return df

# --- 시계열 변수 추가 (lag, rolling mean/std) ---
def add_time_series_features(group):
    group = group.sort_values('영업일자_dt').copy()
    group['lag_1'] = group['매출수량'].shift(1)
    group['lag_7'] = group['매출수량'].shift(7)
    group['rolling_mean_7'] = group['매출수량'].shift(1).rolling(7).mean()
    group['rolling_std_7'] = group['매출수량'].shift(1).rolling(7).std()
    return group

# --- SMAPE 계산 함수 ---
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

# --- 학습 ---
def train_xgboost_models(train_df):
    train_df = add_date_features(train_df)
    trained_models = {}

    for store_menu, group in tqdm(train_df.groupby('영업장명_메뉴명'), desc='Training XGBoost'):
        group = add_time_series_features(group)
        group = group.dropna()

        if len(group) < LOOKBACK + PREDICT:
            continue

        features, targets = [], []
        feature_cols = ['매출수량', '요일', '주말여부', '공휴일여부', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']

        for i in range(len(group) - LOOKBACK - PREDICT + 1):
            past = group.iloc[i:i+LOOKBACK][feature_cols].values.flatten()
            future = group.iloc[i+LOOKBACK:i+LOOKBACK+PREDICT]['매출수량'].values
            features.append(past)
            targets.append(future)

        X_train = np.array(features)
        Y_train = np.array(targets)

        models = []
        for i in range(PREDICT):
            model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.03, random_state=42, n_jobs=-1)
            model.fit(X_train, Y_train[:, i])
            models.append(model)

        last = group.iloc[-LOOKBACK:][feature_cols].values.flatten()
        trained_models[store_menu] = {'models': models, 'last_input': last}

    return trained_models

# --- 예측 ---
def predict_xgboost(test_df, trained_models, test_prefix):
    test_df = add_date_features(test_df)
    results = []

    for store_menu, store_test in test_df.groupby('영업장명_메뉴명'):
        if store_menu not in trained_models:
            continue

        models = trained_models[store_menu]['models']
        input_x = trained_models[store_menu]['last_input'].reshape(1, -1)

        preds = []
        for model in models:
            pred = model.predict(input_x)[0]
            preds.append(max(pred, 0))

        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, preds):
            results.append({
                '영업일자': d,
                '영업장명_메뉴명': store_menu,
                '매출수량': val
            })

    return pd.DataFrame(results)

# --- 제출 포맷 변환 ---
def convert_to_submission_format(pred_df, sample_submission):
    pred_dict = dict(zip(zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']), pred_df['매출수량']))
    final_df = sample_submission.copy()
    for idx in final_df.index:
        date = final_df.loc[idx, '영업일자']
        for col in final_df.columns[1:]:
            final_df.loc[idx, col] = pred_dict.get((date, col), 0)
    return final_df

# --- 평가 (SMAPE) ---
def evaluate_smape(pred_df, test_dir='./test'):
    actuals = []
    preds = []

    for path in sorted(glob.glob(f'{test_dir}/TEST_*.csv')):
        test_df = pd.read_csv(path)
        test_prefix = re.search(r'(TEST_\d+)', os.path.basename(path)).group(1)

        test_df = add_date_features(test_df)
        test_grouped = test_df.groupby('영업장명_메뉴명')['매출수량'].apply(list)

        for store_menu, actual_vals in test_grouped.items():
            for i in range(PREDICT):
                key = (f"{test_prefix}+{i+1}일", store_menu)
                pred = pred_df[pred_df['영업일자'] == key[0]]
                val = pred[pred['영업장명_메뉴명'] == key[1]]['매출수량'].values
                if len(val) == 1:
                    preds.append(val[0])
                    actuals.append(actual_vals[i])

    return smape(np.array(actuals), np.array(preds))

# --- 메인 실행 ---
if __name__ == "__main__":
    train = pd.read_csv('./train/train.csv')
    trained_models = train_xgboost_models(train)

    all_preds = []
    test_files = sorted(glob.glob('./test/TEST_*.csv'))
    for path in test_files:
        test_df = pd.read_csv(path)
        filename = os.path.basename(path)
        test_prefix = re.search(r'(TEST_\d+)', filename).group(1)
        pred_df = predict_xgboost(test_df, trained_models, test_prefix)
        all_preds.append(pred_df)

    full_pred_df = pd.concat(all_preds, ignore_index=True)

    # 평가
    smape_score = evaluate_smape(full_pred_df)
    print(f"\n✅ SMAPE: {smape_score:.4f}")

    # 제출 파일 저장
    sample_submission = pd.read_csv('./sample_submission.csv')
    submission = convert_to_submission_format(full_pred_df, sample_submission)
    submission.to_csv('xgboost_submission_4.csv', index=False, encoding='utf-8-sig')

#✅ SMAPE: 138.4230