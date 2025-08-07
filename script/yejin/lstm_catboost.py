import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    df = df.sort_values(['영업장명_메뉴명', '영업일자'])
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).std()

    df['매출수량'] = df['매출수량'].clip(lower=0)
    return df


# LSTM Model 정의
class SalesDataset(Dataset):
    def __init__(self, seq_x, seq_y=None):
        self.seq_x = seq_x
        self.seq_y = seq_y
    def __len__(self):
        return len(self.seq_x)
    def __getitem__(self, idx):
        x = torch.tensor(self.seq_x[idx], dtype=torch.float32)
        if self.seq_y is not None:
            y = torch.tensor(self.seq_y[idx], dtype=torch.float32)
            return x, y
        else:
            return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# 데이터 로드 및 전처리
train = pd.read_csv('./train/train.csv')
train['영업일자'] = pd.to_datetime(train['영업일자'])
train['day_of_week'] = train['영업일자'].dt.weekday
train['is_weekend'] = train['day_of_week'].isin([5,6]).astype(int)
train['month'] = train['영업일자'].dt.month
train['is_holiday'] = train['영업일자'].apply(is_korean_holiday)
train['매출수량'] = train['매출수량'].clip(lower=0)

# 피처 생성 (CatBoost용)
train_cat = create_features(train).dropna()
cat_features = ['영업장명_메뉴명']
features = [col for col in train_cat.columns if col not in ['영업일자', '매출수량']]

# LSTM용 시계열 데이터 생성
seq_len, pred_len = 28, 7
seq_x_list, seq_y_list = [], []
for _, df_menu in train.groupby('영업장명_메뉴명'):
    df_menu = df_menu.sort_values('영업일자')
    arr = df_menu[['매출수량', 'day_of_week', 'is_weekend', 'month', 'is_holiday']].values
    for i in range(len(arr) - seq_len - pred_len + 1):
        seq_x_list.append(arr[i:i+seq_len])
        seq_y_list.append(arr[i+seq_len:i+seq_len+pred_len, 0])

seq_x_train, seq_x_val, seq_y_train, seq_y_val = train_test_split(
    seq_x_list, seq_y_list, test_size=0.1, random_state=42
)

train_dataset = SalesDataset(seq_x_train, seq_y_train)
val_dataset = SalesDataset(seq_x_val, seq_y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# LSTM 모델 학습 및 최적 하이퍼파라미터 탐색
lstm_params_list = [
    {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'batch_size': 64},
    {'hidden_dim': 128, 'num_layers': 2, 'lr': 0.001, 'batch_size': 64},
    {'hidden_dim': 64, 'num_layers': 3, 'lr': 0.0005, 'batch_size': 64},
]

best_loss = float('inf')
best_params = None
best_lstm_model = None

for params in lstm_params_list:
    model = LSTMModel(input_dim=5, hidden_dim=params['hidden_dim'], num_layers=params['num_layers']).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    train_ds = SalesDataset(seq_x_train, seq_y_train)
    train_loader_tmp = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader_tmp:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader_tmp)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"[LSTM Params {params}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_params = params
        best_lstm_model = model

print(f"Best LSTM Params: {best_params}")


# LSTM 최적 파라미터로 학습
lstm_model = LSTMModel(input_dim=5, hidden_dim=best_params['hidden_dim'], num_layers=best_params['num_layers']).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=best_params['lr'])

patience = 7
best_val_loss = float('inf')
wait = 0

for epoch in range(50):
    lstm_model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = lstm_model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    lstm_model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = lstm_model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        os.makedirs("models", exist_ok=True)
        torch.save(lstm_model.state_dict(), "models/best_lstm_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

print("LSTM 최적 모델 저장 완료")


# CatBoost 하이퍼파라미터 튜닝

# train/validation 분리 (8:2)
train_pool = Pool(train_cat[features], label=train_cat['매출수량'], cat_features=cat_features)

# train/validation 나누기
train_idx, val_idx = train_test_split(train_cat.index, test_size=0.2, random_state=42, shuffle=True)
train_pool = Pool(train_cat.loc[train_idx, features], label=train_cat.loc[train_idx, '매출수량'], cat_features=cat_features)
val_pool = Pool(train_cat.loc[val_idx, features], label=train_cat.loc[val_idx, '매출수량'], cat_features=cat_features)

cat_params_list = [
    {'iterations': 300, 'depth': 6, 'learning_rate': 0.1},
    {'iterations': 500, 'depth': 6, 'learning_rate': 0.05},
    {'iterations': 400, 'depth': 5, 'learning_rate': 0.07},
]

best_cat_loss = float('inf')
best_cat_params = None
best_cat_model = None

for params in cat_params_list:
    model = CatBoostRegressor(**params, random_seed=42, verbose=100, early_stopping_rounds=50)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=100)
    preds_val = model.predict(val_pool)
    val_loss = np.mean((preds_val - train_cat.loc[val_idx, '매출수량']) ** 2)
    print(f"[CatBoost Params {params}] Validation MSE: {val_loss:.4f}")
    if val_loss < best_cat_loss:
        best_cat_loss = val_loss
        best_cat_params = params
        best_cat_model = model

# 최적 CatBoost 모델 저장
os.makedirs("models", exist_ok=True)
best_cat_model.save_model("models/best_catboost_model.cbm")
print(f"Best CatBoost Params: {best_cat_params}")
print("CatBoost 최적 모델 저장 완료")


# 테스트 데이터 예측 및 앙상블

submission_df = pd.read_csv('./sample_submission.csv')

for path in sorted(glob.glob('./test/*.csv')):
    test_df = pd.read_csv(path)
    
    # CatBoost 예측
    test_cat = create_features(test_df).fillna(0)
    pred_cb = []
    for _, df_menu in test_cat.groupby('영업장명_메뉴명'):
        X_test = df_menu[features].tail(1)
        preds_cat = best_cat_model.predict(X_test)
        pred_cb.append(np.repeat(preds_cat, pred_len))

    # LSTM 예측
    test_df['영업일자'] = pd.to_datetime(test_df['영업일자'])
    test_df['day_of_week'] = test_df['영업일자'].dt.weekday
    test_df['is_weekend'] = test_df['day_of_week'].isin([5,6]).astype(int)
    test_df['month'] = test_df['영업일자'].dt.month
    test_df['is_holiday'] = test_df['영업일자'].apply(is_korean_holiday)
    test_df['매출수량'] = test_df['매출수량'].clip(lower=0)

    pred_lstm = []
    for _, df_menu in test_df.groupby('영업장명_메뉴명'):
        seq_input = df_menu[['매출수량','day_of_week','is_weekend','month','is_holiday']].values[-seq_len:]
        seq_input = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = lstm_model(seq_input).cpu().numpy().flatten()
        pred_lstm.append(pred)

    # 앙상블 (단순 평균)
    final_preds = [(np.array(cb) + np.array(ls)) / 2 for cb, ls in zip(pred_cb, pred_lstm)]

    # 음수 클리핑
    final_preds = [np.clip(pred, a_min=0, a_max=None) for pred in final_preds]

   
    basename = os.path.basename(path).replace('.csv', '')
    for idx, menu in enumerate(test_df['영업장명_메뉴명'].unique()):
        submission_df.loc[submission_df['영업일자'].str.startswith(basename), menu] = final_preds[idx]


submission_df.to_csv('submission_with_earlystopping0807.csv', index=False)