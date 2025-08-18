import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 🎯 최종 권장 전략: LightGBM 28일 윈도우 모델 (Phase 1: Baseline 구축)
# - 대회 규칙(28일 고정 윈도우, Data Leakage 방지)을 엄격히 준수
# - 각 메뉴별로 독립적인 모델을 학습
# - 28일간의 데이터를 기반으로 통계/추세/요일별 피처를 생성
# - 예측할 미래 7일의 각 날짜에 대해 별도의 LightGBM 모델을 학습 (총 7개 모델)
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

class RestaurantDemandPredictor:
    """
    28일 고정 윈도우 제약 하에서의 식음업장 수요 예측 모델
    """
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cols = None

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

    def prepare_training_data(self, full_data, dates):
        """전체 데이터를 28일 윈도우로 분할하여 학습 데이터 생성"""
        X_list, y_list = [], []
        
        for i in range(len(full_data) - 34):  # 28일(input) + 7일(target)
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

    def train_lightgbm_model(self, X_train, y_train, menu_name):
        """7일 예측을 위한 7개의 LightGBM 모델 학습"""
        models, scalers = [], []
        
        for day in range(7):
            y_day = y_train[:, day]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            model = LGBMRegressor(random_state=42)
            model.fit(X_scaled, y_day)
            
            models.append(model)
            scalers.append(scaler)
        
        self.models[menu_name] = models
        self.scalers[menu_name] = scalers

    def predict_7days(self, data_28days, last_date, menu_name):
        """28일 데이터로 다음 7일 예측"""
        features = self.create_28day_features(data_28days, last_date)
        X_pred = pd.DataFrame([features])[self.feature_cols] # 학습 때 사용한 피처 순서 유지
        
        predictions = []
        for day in range(7):
            scaler = self.scalers[menu_name][day]
            model = self.models[menu_name][day]
            
            X_scaled = scaler.transform(X_pred)
            pred = model.predict(X_scaled)[0]
            predictions.append(max(0, pred))
        
        return np.array(predictions)

# ==============================================================================
# Phase 2/3 에서 활용 가능한 Transformer 모델 아키텍처 (참고용)
# ==============================================================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=3, output_dim=7):
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

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("데이터 로드 중...")
    train_df = pd.read_csv('./data/train/train.csv')
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
    train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
    submission_df = pd.read_csv('./data/sample_submission.csv')

    predictor = RestaurantDemandPredictor()
    unique_menus = train_df['영업장명_메뉴명'].unique()
    
    # 1. 메뉴별 모델 학습
    for menu_name in tqdm(unique_menus, desc="메뉴별 모델 학습"):
        menu_df = train_df[train_df['영업장명_메뉴명'] == menu_name].sort_values(by='영업일자')
        
        # 학습 데이터가 충분하지 않으면 건너뛰기
        if len(menu_df) < 35:
            continue
            
        sales_data = menu_df['매출수량'].values
        dates = menu_df['영업일자'].values
        
        X_train, y_train = predictor.prepare_training_data(sales_data, dates)
        predictor.train_lightgbm_model(X_train, y_train, menu_name)

    # 2. Test 파일별 예측
    print("\n예측 생성 중...")
    all_predictions = []
    test_paths = sorted(glob.glob('./data/test/*.csv'))
    
    for path in tqdm(test_paths, desc="Test 파일별 예측"):
        test_file_df = pd.read_csv(path)
        test_file_df['영업일자'] = pd.to_datetime(test_file_df['영업일자'])
        basename = os.path.basename(path).replace('.csv', '')
        
        for menu_name in test_file_df['영업장명_메뉴명'].unique():
            if menu_name not in predictor.models:
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
            
            # 예측
            pred_7days = predictor.predict_7days(input_28days_data, last_date, menu_name)
            
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
        
        output_filename = 'submission_lgbm_28day_window.csv'
        final_submission.to_csv(output_filename, index=False)
        print(f"\n{output_filename} 파일 생성 완료")
    else:
        print("생성된 예측이 없습니다.")

    print("\n=== 🏆 28일 고정 윈도우 LightGBM 모델 ===")
    print("✅ 대회 규칙을 준수하는 맞춤형 모델 구현 완료")
    print("✅ 각 샘플은 독립적으로 28일 데이터만 사용하여 예측 수행")
    print("✅ 도메인 지식(요일, 공휴일)만 활용")
