import os
import random
import glob
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# --- 고정 시드 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

# --- 공휴일 2023~2024 ---
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

# --- 고급 피처 엔지니어링 ---
def create_advanced_features(df):
    """데이터 분석 리포트 기반 고급 피처 생성"""
    df = df.copy()
    
    # 날짜 파싱
    df['영업일자_dt'] = pd.to_datetime(df['영업일자'])
    
    # 기본 시간 피처
    df['년'] = df['영업일자_dt'].dt.year
    df['월'] = df['영업일자_dt'].dt.month
    df['일'] = df['영업일자_dt'].dt.day
    df['요일'] = df['영업일자_dt'].dt.dayofweek
    df['주차'] = df['영업일자_dt'].dt.isocalendar().week
    df['분기'] = df['영업일자_dt'].dt.quarter
    
    # 계절성 피처 (분석 리포트 기반)
    df['겨울'] = df['월'].isin([12, 1, 2]).astype(int)
    df['봄'] = df['월'].isin([3, 4, 5]).astype(int)
    df['여름'] = df['월'].isin([6, 7, 8]).astype(int)
    df['가을'] = df['월'].isin([9, 10, 11]).astype(int)
    
    # 특별한 월 패턴 (3월 급감, 1월 최고)
    df['3월_특이'] = (df['월'] == 3).astype(int)
    df['1월_최고'] = (df['월'] == 1).astype(int)
    
    # 주말/공휴일
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['공휴일여부'] = df['영업일자_dt'].isin(pd.to_datetime(holidays_2023_2024)).astype(int)
    
    # 영업장별 분류
    df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
    df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:]
    df['메뉴명'] = df['메뉴명'].apply(lambda x: '_'.join(x) if x else '')
    
    # 영업장별 특성 (분석 리포트 기반)
    df['포레스트릿'] = (df['영업장명'] == '포레스트릿').astype(int)
    df['카페테리아'] = (df['영업장명'] == '카페테리아').astype(int)
    df['화담숲주막'] = (df['영업장명'] == '화담숲주막').astype(int)
    df['담하'] = (df['영업장명'] == '담하').astype(int)
    df['미라시아'] = (df['영업장명'] == '미라시아').astype(int)
    
    # 메뉴 카테고리 분류
    df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김', na=False).astype(int)
    df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼', na=False).astype(int)
    df['주류류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인', na=False).astype(int)
    df['한식류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥', na=False).astype(int)
    df['양식류'] = df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또', na=False).astype(int)
    df['단체메뉴'] = df['메뉴명'].str.contains('단체|패키지|세트', na=False).astype(int)
    df['대여료'] = df['메뉴명'].str.contains('대여료|이용료', na=False).astype(int)
    
    # 시간대별 특성
    df['브런치_시간'] = ((df['영업장명'] == '미라시아') & (df['메뉴명'].str.contains('브런치', na=False))).astype(int)
    
    # 지연 피처 (과거 데이터 활용)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'매출수량_lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
    
    # 이동평균 피처
    for window in [3, 7, 14, 30]:
        df[f'매출수량_ma_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
    
    # 표준편차 피처 (변동성)
    for window in [7, 14, 30]:
        df[f'매출수량_std_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
    
    # 영업장별 메뉴별 통계
    menu_stats = df.groupby('영업장명_메뉴명')['매출수량'].agg(['mean', 'std', 'max', 'min']).reset_index()
    menu_stats.columns = ['영업장명_메뉴명', 'menu_mean', 'menu_std', 'menu_max', 'menu_min']
    df = df.merge(menu_stats, on='영업장명_메뉴명', how='left')
    
    # 영업장별 통계
    store_stats = df.groupby('영업장명')['매출수량'].agg(['mean', 'std']).reset_index()
    store_stats.columns = ['영업장명', 'store_mean', 'store_std']
    df = df.merge(store_stats, on='영업장명', how='left')
    
    return df

# --- 앙상블 모델 학습 함수 ---
def train_ensemble_model(train_df, target_col='매출수량'):
    """앙상블 모델 학습 (CatBoost + LightGBM + XGBoost)"""
    
    # 피처 생성
    train_df = create_advanced_features(train_df)
    
    # 카테고리형 변수 인코딩
    categorical_features = ['영업장명', '메뉴명', '영업장명_메뉴명']
    label_encoders = {}
    
    for col in categorical_features:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[f'{col}_encoded'] = le.fit_transform(train_df[col].astype(str))
            label_encoders[col] = le
    
    # 수치형 피처 선택
    numeric_features = [
        '년', '월', '일', '요일', '주차', '분기',
        '겨울', '봄', '여름', '가을', '3월_특이', '1월_최고',
        '주말여부', '공휴일여부',
        '포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아',
        '분식류', '음료류', '주류류', '한식류', '양식류', '단체메뉴', '대여료', '브런치_시간',
        '매출수량_lag_1', '매출수량_lag_2', '매출수량_lag_3', '매출수량_lag_7', '매출수량_lag_14', '매출수량_lag_30',
        '매출수량_ma_3', '매출수량_ma_7', '매출수량_ma_14', '매출수량_ma_30',
        '매출수량_std_7', '매출수량_std_14', '매출수량_std_30',
        'menu_mean', 'menu_std', 'menu_max', 'menu_min',
        'store_mean', 'store_std'
    ]
    
    # 카테고리형 피처
    categorical_features_encoded = [f'{col}_encoded' for col in categorical_features if col in train_df.columns]
    
    # 최종 피처 리스트
    feature_columns = numeric_features + categorical_features_encoded
    
    # 결측치 처리
    for col in feature_columns:
        if col in train_df.columns:
            if col in categorical_features_encoded:
                train_df[col] = train_df[col].fillna(-1)
            else:
                train_df[col] = train_df[col].fillna(0)
    
    # 시계열 분할
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 앙상블 모델들
    models = {}
    
    # 1. CatBoost
    print("CatBoost 모델 학습 중...")
    catboost_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    # 2. LightGBM
    print("LightGBM 모델 학습 중...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # 3. XGBoost
    print("XGBoost 모델 학습 중...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    # 교차 검증 및 모델 학습
    cv_scores = []
    for train_idx, val_idx in tscv.split(train_df):
        X_train = train_df.iloc[train_idx][feature_columns]
        y_train = train_df.iloc[train_idx][target_col]
        X_val = train_df.iloc[val_idx][feature_columns]
        y_val = train_df.iloc[val_idx][target_col]
        
        # CatBoost 학습
        catboost_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=categorical_features_encoded,
            use_best_model=True
        )
        
        # LightGBM 학습
        lgb_model.fit(X_train, y_train)
        
        # XGBoost 학습
        xgb_model.fit(X_train, y_train)
        
        # 앙상블 예측
        catboost_pred = catboost_model.predict(X_val)
        lgb_pred = lgb_model.predict(X_val)
        xgb_pred = xgb_model.predict(X_val)
        
        # 가중 평균 (CatBoost: 0.5, LightGBM: 0.3, XGBoost: 0.2)
        ensemble_pred = 0.5 * catboost_pred + 0.3 * lgb_pred + 0.2 * xgb_pred
        
        rmse = np.sqrt(np.mean((y_val - ensemble_pred) ** 2))
        cv_scores.append(rmse)
        print(f"CV Score: {rmse:.4f}")
    
    print(f"Average CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # 전체 데이터로 최종 모델 학습
    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    
    # 최종 모델들 학습
    final_catboost = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    final_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    final_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    final_catboost.fit(X_train, y_train, cat_features=categorical_features_encoded)
    final_lgb.fit(X_train, y_train)
    final_xgb.fit(X_train, y_train)
    
    models = {
        'catboost': final_catboost,
        'lightgbm': final_lgb,
        'xgboost': final_xgb
    }
    
    return models, label_encoders, feature_columns

# --- 앙상블 예측 함수 ---
def predict_ensemble(test_df, models, label_encoders, feature_columns):
    """앙상블 모델로 매출 예측"""
    
    # 피처 생성 (테스트 데이터의 매출수량을 사용하여 피처 생성)
    test_df = create_advanced_features(test_df)
    
    # 카테고리형 변수 인코딩
    for col, le in label_encoders.items():
        if col in test_df.columns:
            test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
    
    # 결측치 처리
    for col in feature_columns:
        if col in test_df.columns:
            if col.endswith('_encoded'):
                test_df[col] = test_df[col].fillna(-1)
            else:
                test_df[col] = test_df[col].fillna(0)
    
    # 각 모델별 예측
    catboost_pred = models['catboost'].predict(test_df[feature_columns])
    lgb_pred = models['lightgbm'].predict(test_df[feature_columns])
    xgb_pred = models['xgboost'].predict(test_df[feature_columns])
    
    # 앙상블 예측 (가중 평균)
    ensemble_pred = 0.5 * catboost_pred + 0.3 * lgb_pred + 0.2 * xgb_pred
    
    # 음수 예측값을 0으로 조정
    ensemble_pred = np.maximum(ensemble_pred, 0)
    
    return ensemble_pred

# --- 피처 중요도 분석 ---
def analyze_feature_importance(models, feature_columns):
    """피처 중요도 분석"""
    print("\n=== 피처 중요도 분석 ===")
    
    # CatBoost 피처 중요도
    catboost_importance = models['catboost'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'catboost_importance': catboost_importance
    })
    
    # LightGBM 피처 중요도
    lgb_importance = models['lightgbm'].feature_importances_
    feature_importance_df['lightgbm_importance'] = lgb_importance
    
    # XGBoost 피처 중요도
    xgb_importance = models['xgboost'].feature_importances_
    feature_importance_df['xgboost_importance'] = xgb_importance
    
    # 평균 중요도
    feature_importance_df['avg_importance'] = (
        feature_importance_df['catboost_importance'] * 0.5 +
        feature_importance_df['lightgbm_importance'] * 0.3 +
        feature_importance_df['xgboost_importance'] * 0.2
    )
    
    # 상위 20개 피처 출력
    top_features = feature_importance_df.nlargest(20, 'avg_importance')
    print("상위 20개 중요 피처:")
    for idx, row in top_features.iterrows():
        print(f"{row['feature']}: {row['avg_importance']:.4f}")
    
    return feature_importance_df

# --- 제출 파일 변환 함수 (이전 버전) ---
def convert_to_submission_format(pred_df, sample_submission):
    # (영업일자, 영업장명_메뉴명) → 예측값 딕셔너리 생성
    def convert_to_integer(value):
        if value < 0:
            return 0
        return int(round(value))

    pred_dict = dict(zip(zip(pred_df['영업일자'].astype(str), pred_df['영업장명_메뉴명'].astype(str)), pred_df['매출수량']))
    final_df = sample_submission.copy()
    for idx in final_df.index:
        date = str(final_df.loc[idx, '영업일자'])
        for col in final_df.columns[1:]:
            val = pred_dict.get((date, str(col)), 0)
            final_df.loc[idx, col] = convert_to_integer(val)
    return final_df

# --- 메인 실행 ---
if __name__ == "__main__":
    print("=== 리조트 식음업장 메뉴별 수요 예측 앙상블 모델 (Baseline 스타일) ===")

    # 1. 데이터 로드
    print("데이터 로드 중...")
    train = pd.read_csv('./data/train/train.csv')
    print(f"훈련 데이터: {train.shape}")

    # 2. 앙상블 모델 학습
    print("앙상블 모델 학습 중...")
    models, label_encoders, feature_columns = train_ensemble_model(train)

    # 3. 테스트 데이터 예측 (모든 TEST_*.csv 순회)
    print("테스트 데이터 예측 중...")
    all_preds = []
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    for path in test_files:
        test_df = pd.read_csv(path)
        preds = predict_ensemble(test_df, models, label_encoders, feature_columns)
        # 예측 결과 DataFrame 생성 시에만 날짜 변환
        filename = os.path.basename(path)
        test_prefix = re.search(r'(TEST_\d+)', filename).group(1)
        base_date = pd.to_datetime(test_df['영업일자'].iloc[0])
        converted_dates = test_df['영업일자'].apply(
            lambda x: f"{test_prefix}+{(pd.to_datetime(x) - base_date).days + 1}일"
        )
        pred_df = pd.DataFrame({
            '영업일자': converted_dates,
            '영업장명_메뉴명': test_df['영업장명_메뉴명'],
            '매출수량': preds
        })
        all_preds.append(pred_df)
    full_pred_df = pd.concat(all_preds, ignore_index=True)

    # 4. 샘플 제출 파일 기준 변환
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    submission = convert_to_submission_format(full_pred_df, sample_submission)

    # 5. 결과 저장
    submission.to_csv('ensemble_improved_submission.csv', index=False, encoding='utf-8-sig')
    print("예측 완료! 결과가 'ensemble_improved_submission.csv'에 저장되었습니다.")
