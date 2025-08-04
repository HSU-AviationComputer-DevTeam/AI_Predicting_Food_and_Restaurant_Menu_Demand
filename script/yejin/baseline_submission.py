import os
import random
import glob
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from tqdm import tqdm

# --- 고정 시드 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

# --- 하이퍼파라미터 ---
LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 16, 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 한국 공휴일 2023~2024 예시 ---
holidays_2023_2024 = [
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03", 
    "2023-10-09", "2023-12-25",

    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", 
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15", 
    "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", 
    "2024-10-03", "2024-10-09", "2024-12-25",
]
holidays = pd.to_datetime(holidays_2023_2024)

# --- 도메인 피처 추가 함수 ---
def add_date_features(df):
    # 날짜 정리: 평가 데이터는 TEST_00+N일 형식, train은 yyyy-mm-dd 형식
    try:
        df['영업일자_dt'] = pd.to_datetime(df['영업일자'])
    except Exception:
        df['영업일자_dt'] = pd.to_datetime(df['영업일자'].str.replace(r'TEST_\d+\+', '', regex=True), errors='coerce')

    df['요일'] = df['영업일자_dt'].dt.dayofweek
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['공휴일여부'] = df['영업일자_dt'].isin(holidays).astype(int)
    return df

# --- 모델 정의 ---
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=PREDICT):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 학습 함수 ---
def train_lstm(train_df):
    trained_models = {}
    train_df = add_date_features(train_df)
    features = ['매출수량', '요일', '주말여부', '공휴일여부']

    for store_menu, group in tqdm(train_df.groupby('영업장명_메뉴명'), desc='Training LSTM'):
        store_train = group.sort_values('영업일자').copy()
        if len(store_train) < LOOKBACK + PREDICT:
            continue

        scaler = MinMaxScaler()
        store_train[features] = scaler.fit_transform(store_train[features])
        train_vals = store_train[features].values

        X_train, y_train = [], []
        for i in range(len(train_vals) - LOOKBACK - PREDICT + 1):
            X_train.append(train_vals[i:i+LOOKBACK])
            y_train.append(train_vals[i+LOOKBACK:i+LOOKBACK+PREDICT, 0])  # 매출수량만 타겟

        X_train = torch.tensor(X_train).float().to(DEVICE)
        y_train = torch.tensor(y_train).float().to(DEVICE)

        model = MultiOutputLSTM(input_dim=len(features), output_dim=PREDICT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(EPOCHS):
            idx = torch.randperm(len(X_train))
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_idx = idx[i:i+BATCH_SIZE]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                output = model(X_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        trained_models[store_menu] = {
            'model': model.eval(),
            'scaler': scaler,
            'last_sequence': train_vals[-LOOKBACK:]
        }
    return trained_models

# --- 예측 함수 ---
def predict_lstm(test_df, trained_models, test_prefix):
    results = []
    test_df = add_date_features(test_df)
    features = ['매출수량', '요일', '주말여부', '공휴일여부']

    for store_menu, store_test in test_df.groupby('영업장명_메뉴명'):
        if store_menu not in trained_models:
            continue

        model = trained_models[store_menu]['model']
        scaler = trained_models[store_menu]['scaler']

        store_test_sorted = store_test.sort_values('영업일자')
        recent_vals = store_test_sorted[features].values[-LOOKBACK:]
        if len(recent_vals) < LOOKBACK:
            continue

        recent_vals_scaled = scaler.transform(recent_vals)
        x_input = torch.tensor([recent_vals_scaled]).float().to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x_input).squeeze().cpu().numpy()

        restored = []
        for i in range(PREDICT):
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = pred_scaled[i]  # 매출수량만 역변환
            restored_val = scaler.inverse_transform(dummy)[0, 0]
            restored.append(max(restored_val, 0))

        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, restored):
            results.append({
                '영업일자': d,
                '영업장명_메뉴명': store_menu,
                '매출수량': val
            })

    return pd.DataFrame(results)

# --- 메인 ---
if __name__ == "__main__":
    train = pd.read_csv('./train/train.csv')
    trained_models = train_lstm(train)

    all_preds = []
    test_files = sorted(glob.glob('./test/TEST_*.csv'))
    for path in test_files:
        test_df = pd.read_csv(path)
        filename = os.path.basename(path)
        test_prefix = re.search(r'(TEST_\d+)', filename).group(1)
        pred_df = predict_lstm(test_df, trained_models, test_prefix)
        all_preds.append(pred_df)

    full_pred_df = pd.concat(all_preds, ignore_index=True)

    sample_submission = pd.read_csv('./sample_submission.csv')
    def convert_to_submission_format(pred_df, sample_submission):
        pred_dict = dict(zip(zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']), pred_df['매출수량']))
        final_df = sample_submission.copy()
        for idx in final_df.index:
            date = final_df.loc[idx, '영업일자']
            for col in final_df.columns[1:]:
                final_df.loc[idx, col] = pred_dict.get((date, col), 0)
        return final_df

    submission = convert_to_submission_format(full_pred_df, sample_submission)
    submission.to_csv('improved_submission.csv', index=False, encoding='utf-8-sig')
