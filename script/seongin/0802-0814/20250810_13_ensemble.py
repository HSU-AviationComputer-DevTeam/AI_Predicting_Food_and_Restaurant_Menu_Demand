import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 🎯 최종 권장 전략: LightGBM + Transformer 앙상블 모델 (Phase 2: 모델 개선)
# - LightGBM(피처 기반)과 Transformer(시퀀스 기반) 모델을 모두 학습
# - 예측 시, 두 모델의 결과를 50:50으로 단순 평균하여 앙상블
# - 대회 규칙(28일 고정 윈도우, Data Leakage 방지)을 엄격히 준수
# ==============================================================================


# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# 공휴일 리스트 (도메인 지식)
holiday_dates = pd.to_datetime([
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03",
    "2023-10-09", "2023-12-25", "2024-01-01", "2024-02-09", "2024-02-10",
    "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-05",
    "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
    "2024-09-17", "2024-09-18", "2024-10-03", "2024-10-09", "2024-12-25"
])

def is_korean_holiday(date):
    return int(date in holiday_dates)

# ==============================================================================
# Phase 2/3 에서 활용 가능한 Transformer 모델 아키텍처
# ==============================================================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, output_dim=7):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        pe = torch.zeros(28, d_model)
        position = torch.arange(0, 28).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 28, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoding.to(x.device)
        encoded = self.transformer(x)
        output = self.output_projection(encoded.view(encoded.size(0), -1))
        return output

class RestaurantDemandPredictor:
    """
    LightGBM과 Transformer 앙상블을 위한 수요 예측 모델
    """
    def __init__(self, device):
        self.lgbm_models = {}
        self.lgbm_scalers = {}
        self.transformer_models = {}
        self.transformer_scalers = {}
        self.feature_cols = None
        self.device = device

    def create_28day_features(self, data_28days, last_date):
        """28일 데이터로부터 feature 추출 (Data Leakage 방지)"""
        features = {}
        data_28days = np.array(data_28days)

        # 1. 기본 통계량
        features['mean_sales'] = np.mean(data_28days)
        features['std_sales'] = np.std(data_28days)
        features['median_sales'] = np.median(data_28days)
        features['min_sales'] = np.min(data_28days)
        features['max_sales'] = np.max(data_28days)
        
        # 2. 주별 패턴 (4주간)
        for week in range(4):
            week_data = data_28days[week*7:(week+1)*7]
            features[f'week_{week}_mean'] = np.mean(week_data)

        # 3. 최근 경향성
        features['last_7day_mean'] = np.mean(data_28days[-7:])
        features['recent_trend'] = np.mean(data_28days[-7:]) - np.mean(data_28days[-14:-7])

        # 4. 도메인 지식 (추론 시점에서 알 수 있는 정보만)
        # 다음 7일간의 요일, 주말, 공휴일 정보
        for i in range(1, 8):
            pred_date = last_date + pd.Timedelta(days=i)
            features[f'pred_day_{i}_weekday'] = pred_date.weekday()
            features[f'pred_day_{i}_is_weekend'] = 1 if pred_date.weekday() >= 5 else 0
            features[f'pred_day_{i}_is_holiday'] = is_korean_holiday(pred_date)
            
        return features

    def prepare_lgbm_training_data(self, full_data, dates):
        """LGBM 학습을 위해 28일 윈도우로 피처 데이터 생성"""
        X_list, y_list = [], []
        
        for i in range(len(full_data) - 34):
            input_data = full_data[i:i+28]
            target_data = full_data[i+28:i+35]
            last_date = dates[i+27]
            
            features = self.create_28day_features(input_data, last_date)
            X_list.append(features)
            y_list.append(target_data)
        
        X_df = pd.DataFrame(X_list)
        if self.feature_cols is None:
            self.feature_cols = X_df.columns
        return X_df, np.array(y_list)

    def prepare_transformer_training_data(self, full_data):
        """Transformer 학습을 위해 28일 시퀀스 데이터 생성"""
        X_seq, y_seq = [], []
        for i in range(len(full_data) - 34):
            X_seq.append(full_data[i:i+28])
            y_seq.append(full_data[i+28:i+35])
        return np.array(X_seq), np.array(y_seq)

    def train_lightgbm_model(self, X_train, y_train, menu_name):
        """7일 예측을 위한 7개의 LightGBM 모델 학습"""
        models, scalers = [], []
        for day in range(7):
            y_day = y_train[:, day]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model = LGBMRegressor(random_state=42, verbose=-1)
            model.fit(X_scaled, y_day)
            models.append(model)
            scalers.append(scaler)
        self.lgbm_models[menu_name] = models
        self.lgbm_scalers[menu_name] = scalers

    def train_transformer_model(self, X_train_seq, y_train_seq, menu_name, epochs=10, batch_size=16):
        """Transformer 모델 학습"""
        scaler = StandardScaler()
        # 시퀀스 데이터를 1D로 펼쳐서 스케일러 학습 후 다시 2D로 복원
        X_train_reshaped = X_train_seq.reshape(-1, 1)
        scaler.fit(X_train_reshaped)
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train_seq.shape)

        X_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(-1).to(self.device)
        y_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = TransformerModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        self.transformer_models[menu_name] = model
        self.transformer_scalers[menu_name] = scaler

    def predict_7days_ensemble(self, data_28days, last_date, menu_name):
        """LGBM과 Transformer 예측 결과를 앙상블하여 7일 예측"""
        # 1. LightGBM 예측
        features = self.create_28day_features(data_28days, last_date)
        X_pred_lgbm = pd.DataFrame([features])[self.feature_cols]
        lgbm_preds = []
        for day in range(7):
            scaler = self.lgbm_scalers[menu_name][day]
            model = self.lgbm_models[menu_name][day]
            X_scaled = scaler.transform(X_pred_lgbm)
            pred = model.predict(X_scaled)[0]
            lgbm_preds.append(max(0, pred))
        
        # 2. Transformer 예측
        scaler = self.transformer_scalers[menu_name]
        model = self.transformer_models[menu_name]
        model.eval()
        
        input_scaled = scaler.transform(np.array(data_28days).reshape(-1, 1)).reshape(1, 28, 1)
        input_tensor = torch.FloatTensor(input_scaled).to(self.device)
        
        with torch.no_grad():
            transformer_preds_raw = model(input_tensor).squeeze().cpu().numpy()
        
        transformer_preds = np.maximum(0, transformer_preds_raw)

        # 3. 앙상블 (단순 평균)
        ensemble_preds = (np.array(lgbm_preds) + transformer_preds) / 2.0
        
        return ensemble_preds


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    print("데이터 로드 중...")
    train_df = pd.read_csv('./data/train/train.csv')
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
    train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
    submission_df = pd.read_csv('./data/sample_submission.csv')

    predictor = RestaurantDemandPredictor(device)
    unique_menus = train_df['영업장명_메뉴명'].unique()
    
    # 1. 메뉴별 모델 학습
    for menu_name in tqdm(unique_menus, desc="메뉴별 모델 학습 (LGBM+Transformer)"):
        menu_df = train_df[train_df['영업장명_메뉴명'] == menu_name].sort_values(by='영업일자')
        
        if len(menu_df) < 35: continue
            
        sales_data = menu_df['매출수량'].values
        dates = menu_df['영업일자'].values
        
        # LGBM 학습
        X_train_lgbm, y_train_lgbm = predictor.prepare_lgbm_training_data(sales_data, dates)
        predictor.train_lightgbm_model(X_train_lgbm, y_train_lgbm, menu_name)

        # Transformer 학습
        X_train_tf, y_train_tf = predictor.prepare_transformer_training_data(sales_data)
        predictor.train_transformer_model(X_train_tf, y_train_tf, menu_name, epochs=15) # 에포크 증가

    # 2. Test 파일별 예측
    print("\n앙상블 예측 생성 중...")
    all_predictions = []
    test_paths = sorted(glob.glob('./data/test/*.csv'))
    
    for path in tqdm(test_paths, desc="Test 파일별 예측"):
        test_file_df = pd.read_csv(path)
        test_file_df['영업일자'] = pd.to_datetime(test_file_df['영업일자'])
        basename = os.path.basename(path).replace('.csv', '')
        
        for menu_name in test_file_df['영업장명_메뉴명'].unique():
            if menu_name not in predictor.lgbm_models or menu_name not in predictor.transformer_models:
                continue

            # 28일 입력 데이터 구성 (train 끝부분 + test)
            train_menu_df = train_df[train_df['영업장명_메뉴명'] == menu_name]
            
            # test 파일의 첫 날짜와 train 데이터의 마지막 날짜 사이의 gap 계산
            test_start_date = test_file_df['영업일자'].min()
            train_last_date = train_menu_df['영업일자'].max()
            
            # 필요한 과거 데이터 일 수 계산
            days_needed_from_train = 28 - len(test_file_df)
            if days_needed_from_train <= 0:
                 # test 파일만으로 28일이 채워지는 경우
                 context_df = test_file_df.sort_values(by='영업일자').tail(28)
            else:
                # train 데이터와 test 데이터를 합쳐 28일 구성
                historical_data = train_menu_df.sort_values(by='영업일자').tail(days_needed_from_train)
                context_df = pd.concat([historical_data, test_file_df])

            input_28days_data = context_df['매출수량'].values
            last_date = context_df['영업일자'].max()
            
            # 앙상블 예측
            pred_7days = predictor.predict_7days_ensemble(input_28days_data, last_date, menu_name)
            
            # 결과 저장
            for i, pred_val in enumerate(pred_7days):
                all_predictions.append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred_val
                })

    # 3. 제출 파일 생성
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
        final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
        final_submission = final_submission.fillna(0)
        final_submission = final_submission[submission_df.columns]
        
        output_filename = 'submission_ensemble_lgbm_tf.csv'
        final_submission.to_csv(output_filename, index=False)
        print(f"\n{output_filename} 파일 생성 완료")
    else:
        print("생성된 예측이 없습니다.")

    print("\n=== 🏆 LGBM + Transformer 앙상블 모델 ===")
    print("✅ LightGBM(피처)과 Transformer(시퀀스) 모델의 예측을 평균하여 앙상블")
    print("✅ 대회 규칙을 준수하는 맞춤형 모델 구현 완료")
