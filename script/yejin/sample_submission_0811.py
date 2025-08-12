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
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Weighted SMAPE
def weighted_smape(y_true, y_pred, menu_names, epsilon=1e-6):
    weights = np.array([2.0 if '담하' in m or '미라시아' in m else 1.0 for m in menu_names])
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon) / 2
    smape_vals = numerator / denominator
    if len(smape_vals) != len(weights):
        print(f"Shape mismatch: smape_vals={len(smape_vals)}, weights={len(weights)}")
        raise ValueError(f"Shape mismatch: smape_vals={len(smape_vals)}, weights={len(weights)}")
    return np.sum(smape_vals * weights) / np.sum(weights) * 100

# 공휴일
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

# monthly avg helper
def add_menu_monthly_avg(df, ref_df):
    ref = ref_df.copy()
    ref['month'] = ref['영업일자'].dt.month
    monthly_avg = (
        ref.groupby(['영업장명_메뉴명', 'month'])['매출수량']
        .mean()
        .reset_index()
        .rename(columns={'매출수량': 'menu_monthly_avg'})
    )
    df = df.copy()
    df['month'] = df['영업일자'].dt.month
    df = df.merge(monthly_avg, on=['영업장명_메뉴명', 'month'], how='left')
    df['menu_monthly_avg'] = df['menu_monthly_avg'].fillna(0)
    return df

# feature creation
def create_features(df, ref_df=None):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['day_of_week'] = df['영업일자'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
    df['week_of_year'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_holiday'] = df['영업일자'].apply(is_korean_holiday)

    if ref_df is not None:
        df = add_menu_monthly_avg(df, ref_df)

    df = df.sort_values(['영업장명_메뉴명', '영업일자'])
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).std()

    df['매출수량'] = df['매출수량'].clip(lower=0)
    return df

# Dataset & Model
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
train['매출수량'] = train['매출수량'].clip(lower=0)

# feature 생성
train_cat_all = create_features(train, ref_df=train)

# drop rows with NaN
train_cat = train_cat_all.dropna().reset_index(drop=False)

cat_features = ['영업장명_메뉴명']
features = [c for c in train_cat.columns if c not in ['영업일자', '매출수량', 'index']]

# sequence settings
seq_len, pred_len = 28, 7
lstm_feature_cols = ['매출수량', 'day_of_week', 'is_weekend', 'month', 'is_holiday', 'menu_monthly_avg', 'lag_1', 'lag_7', 'rolling_mean_7']

# 동일한 train/val 인덱스 생성
train_idx, val_idx = train_test_split(train_cat.index, test_size=0.2, random_state=42, shuffle=True)
train_idx_set = set(train_idx)
val_idx_set = set(val_idx)

# LSTM 시퀀스 생성
seq_x_list = []
seq_y_list = []
seq_target_idx_list = []
seq_menu_list = []

for menu, df_menu in train_cat.groupby('영업장명_메뉴명'):
    df_menu = df_menu.sort_values('영업일자')
    arr = df_menu[lstm_feature_cols].values
    idxs = df_menu.index.tolist()
    for i in range(len(arr) - seq_len - pred_len + 1):
        seq_x_list.append(arr[i:i+seq_len])
        seq_y_list.append(arr[i+seq_len:i+seq_len+pred_len, 0])
        seq_target_idx_list.append(idxs[i+seq_len])
        seq_menu_list.append([menu] * pred_len)

# seq들을 train/val로 나눔
seq_x_train, seq_y_train = [], []
seq_x_val, seq_y_val = [], []
val_seq_indices, val_menu_names = [], []
for x, y, tidx, menus in zip(seq_x_list, seq_y_list, seq_target_idx_list, seq_menu_list):
    if tidx in train_idx_set:
        seq_x_train.append(x)
        seq_y_train.append(y)
    elif tidx in val_idx_set:
        seq_x_val.append(x)
        seq_y_val.append(y)
        val_seq_indices.append(tidx)
        val_menu_names.extend(menus)

# Dataset loaders
train_dataset = SalesDataset(seq_x_train, seq_y_train)
val_dataset = SalesDataset(seq_x_val, seq_y_val)

# CatBoost Optuna 튜닝
def catboost_optuna_search(train_cat, train_idx, val_idx, features, cat_features, n_trials=30):
    train_pool = Pool(train_cat.loc[train_idx, features], label=train_cat.loc[train_idx, '매출수량'], cat_features=cat_features)
    val_pool = Pool(train_cat.loc[val_idx, features], label=train_cat.loc[val_idx, '매출수량'], cat_features=cat_features)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 600),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': False
        }
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        preds_val = model.predict(val_pool)
        menu_names = train_cat.loc[val_idx, '영업장명_메뉴명'].values
        return weighted_smape(train_cat.loc[val_idx, '매출수량'].values, preds_val, menu_names)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params.update({'random_seed': 42, 'early_stopping_rounds': 50, 'verbose': 100})
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(Pool(train_cat.loc[train_idx, features], label=train_cat.loc[train_idx, '매출수량'], cat_features=cat_features),
                   eval_set=Pool(train_cat.loc[val_idx, features], label=train_cat.loc[val_idx, '매출수량'], cat_features=cat_features),
                   use_best_model=True)
    return best_model, best_params

# LSTM Optuna 튜닝
def lstm_optuna_search(seq_x_train, seq_y_train, seq_x_val, seq_y_val, val_menu_names, lstm_feature_cols, device, n_trials=20):
    def objective(trial):
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        model = LSTMModel(input_dim=len(lstm_feature_cols), hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_ds = SalesDataset(seq_x_train, seq_y_train)
        val_ds = SalesDataset(seq_x_val, seq_y_val)
        train_loader_tmp = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader_tmp = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader_tmp):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            if i >= 3:
                break

        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader_tmp:
                x_batch = x_batch.to(device)
                preds = model(x_batch).cpu().numpy()
                preds_list.append(preds)
                trues_list.append(y_batch.numpy())
        if len(preds_list) == 0:
            return float('inf')
        preds_np = np.vstack(preds_list)
        trues_np = np.vstack(trues_list)
        print(f"Shapes: trues_np={trues_np.shape}, preds_np={preds_np.shape}, val_menu_names={len(val_menu_names)}")
        return weighted_smape(trues_np.flatten(), preds_np.flatten(), val_menu_names)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params

# 하이퍼파라미터 탐색 & 학습
best_cat_model, best_cat_params = catboost_optuna_search(train_cat, train_idx, val_idx, features, cat_features, n_trials=30)
os.makedirs("models", exist_ok=True)
best_cat_model.save_model("models/best_catboost_model.cbm")
print("Best CatBoost Params:", best_cat_params)

best_lstm_params = lstm_optuna_search(seq_x_train, seq_y_train, seq_x_val, seq_y_val, val_menu_names, lstm_feature_cols, device, n_trials=20)
print("Best LSTM Params:", best_lstm_params)

# LSTM 최종 학습
lstm_model = LSTMModel(input_dim=len(lstm_feature_cols),
                       hidden_dim=best_lstm_params['hidden_dim'],
                       num_layers=best_lstm_params['num_layers']).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=best_lstm_params['lr'])

train_loader = DataLoader(SalesDataset(seq_x_train, seq_y_train), batch_size=best_lstm_params['batch_size'], shuffle=True)
val_loader = DataLoader(SalesDataset(seq_x_val, seq_y_val), batch_size=best_lstm_params['batch_size'], shuffle=False)

patience = 10
best_val_smape = float('inf')
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
    preds_list, trues_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = lstm_model(x_batch).cpu().numpy()
            preds_list.append(preds)
            trues_list.append(y_batch.cpu().numpy())
    preds_np = np.vstack(preds_list)
    trues_np = np.vstack(trues_list)
    val_smape_score = weighted_smape(trues_np.flatten(), preds_np.flatten(), val_menu_names)

    print(f"[Epoch {epoch+1}] Train Loss (MSE): {avg_train_loss:.4f} | Val Weighted SMAPE: {val_smape_score:.4f}")

    if val_smape_score < best_val_smape:
        best_val_smape = val_smape_score
        wait = 0
        torch.save(lstm_model.state_dict(), "models/best_lstm_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

print("Saved best LSTM with val Weighted SMAPE:", best_val_smape)

# Validation 예측 정렬 및 앙상블 가중치 탐색
val_pool_all = Pool(train_cat.loc[val_idx, features], label=train_cat.loc[val_idx, '매출수량'], cat_features=cat_features)
val_pred_cb_all = best_cat_model.predict(val_pool_all)

if not os.path.exists("models/best_lstm_model.pth"):
    raise FileNotFoundError("LSTM model file not found: models/best_lstm_model.pth")
lstm_model.load_state_dict(torch.load("models/best_lstm_model.pth"))
lstm_model.eval()

val_ds_for_preds = SalesDataset(seq_x_val, seq_y_val)
val_loader_for_preds = DataLoader(val_ds_for_preds, batch_size=best_lstm_params['batch_size'], shuffle=False)

val_pred_lstm_list = []
with torch.no_grad():
    for x_batch, _ in val_loader_for_preds:
        x_batch = x_batch.to(device)
        preds = lstm_model(x_batch).cpu().numpy()
        val_pred_lstm_list.append(preds)
val_pred_lstm = np.vstack(val_pred_lstm_list)

val_seq_idx_array = np.array(val_seq_indices)
val_cb_for_seq = best_cat_model.predict(train_cat.loc[val_seq_idx_array, features])
val_cb_for_seq_expanded = np.repeat(val_cb_for_seq.reshape(-1,1), pred_len, axis=1)

assert val_cb_for_seq_expanded.shape[0] == val_pred_lstm.shape[0], \
    f"Shapes mismatch after alignment: cb {val_cb_for_seq_expanded.shape}, lstm {val_pred_lstm.shape}"

ensemble_weights = np.arange(0.0, 1.01, 0.01)
best_weight = None
best_score = float('inf')
for w_cb in ensemble_weights:
    w_lstm = 1 - w_cb
    combined = w_cb * val_cb_for_seq_expanded + w_lstm * val_pred_lstm
    score = weighted_smape(np.vstack(seq_y_val).flatten(), combined.flatten(), val_menu_names)
    print(f"w_cb={w_cb:.2f} w_lstm={w_lstm:.2f} => Val Weighted SMAPE={score:.6f}")
    if score < best_score:
        best_score = score
        best_weight = w_cb

print("Best ensemble weight:", best_weight, "best Weighted SMAPE:", best_score)

# 테스트셋 예측 및 제출파일 생성
submission_df = pd.read_csv('./sample_submission.csv')
menu_cols = submission_df.columns.drop('영업일자')
test_files = sorted(glob.glob('./test/TEST_*.csv'))

pred_cb_dict = {}
pred_lstm_dict = {}

for file_path in test_files:
    basename = os.path.basename(file_path)
    test_df = pd.read_csv(file_path)
    test_df['영업일자'] = pd.to_datetime(test_df['영업일자'])
    test_df['매출수량'] = test_df['매출수량'].clip(lower=0)
    test_df = test_df.sort_values(['영업장명_메뉴명', '영업일자'])

    print(f"Test file: {basename}, Rows: {test_df.shape[0]}, Unique menus: {test_df['영업장명_메뉴명'].nunique()}")

    test_cat = create_features(test_df, ref_df=train).fillna(0)

    # 디버깅
    print(f"Test file: {basename}, Columns in test_cat: {test_cat.columns.tolist()}")

    # 메뉴 일치성 점검
    test_menus = set(test_cat['영업장명_메뉴명'].unique())
    submission_menus = set(menu_cols)
    missing_menus = submission_menus - test_menus
    if missing_menus:
        print(f"Warning: Menus in submission but not in test data: {missing_menus}")

    # CatBoost (28일 시퀀스 활용)
    for menu, df_menu in test_cat.groupby('영업장명_메뉴명'):
        if menu not in menu_cols:
            continue
        X_test = df_menu[features].tail(seq_len)
        if X_test.shape[0] == 0:
            continue
        preds = []
        for i in range(min(X_test.shape[0], seq_len)):
            pred = best_cat_model.predict(X_test.iloc[[i]])
            preds.append(pred[0])
        pred = np.mean(preds) if preds else 0
        pred_cb_dict[(basename, menu)] = np.repeat(pred, pred_len)

    # LSTM
    for menu, df_menu in test_cat.groupby('영업장명_메뉴명'):
        if menu not in menu_cols:
            continue
        arr = df_menu[lstm_feature_cols].values
        seq_input = arr[-seq_len:]
        if len(seq_input) < seq_len:
            pad_len = seq_len - len(seq_input)
            seq_input = np.vstack([np.zeros((pad_len, len(lstm_feature_cols))), seq_input])
        seq_tensor = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = lstm_model(seq_tensor).cpu().numpy().flatten()
        pred_lstm_dict[(basename, menu)] = pred

# 제출 채우기
for idx, row in submission_df.iterrows():
    date_str = row['영업일자']
    try:
        test_part, daypart = date_str.split('+')
        test_file = test_part
        day_offset = int(daypart.replace('일','')) - 1
    except Exception:
        test_file = None
        day_offset = 0

    if test_file is None:
        for menu in menu_cols:
            submission_df.at[idx, menu] = 0
        continue

    basename = f"{test_file}.csv"

    for menu in menu_cols:
        key = (basename, menu)
        cb = pred_cb_dict.get(key)
        lstm_p = pred_lstm_dict.get(key)
        if (cb is None) and (lstm_p is None):
            val = 0
        elif cb is None:
            val = max(lstm_p[day_offset], 0)
        elif lstm_p is None:
            val = max(cb[day_offset], 0)
        else:
            val7 = best_weight * cb + (1 - best_weight) * lstm_p
            val = float(np.clip(val7[day_offset], 0, None))
        submission_df.at[idx, menu] = val

print(f"pred_cb_dict size: {len(pred_cb_dict)}, pred_lstm_dict size: {len(pred_lstm_dict)}")
print(f"Submission shape: {submission_df.shape}")
submission_df.to_csv('submission_0811_2.csv', index=False)
print("submission_0811_2.csv 생성 완료")