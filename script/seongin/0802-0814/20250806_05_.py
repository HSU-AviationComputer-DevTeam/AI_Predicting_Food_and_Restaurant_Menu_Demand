import os
import random
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
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

# --- 안전한 라벨 인코딩 함수 ---
def safe_label_encode(le, values):
    """새로운 카테고리를 안전하게 처리하는 라벨 인코딩"""
    result = []
    for val in values.astype(str):
        try:
            result.append(le.transform([val])[0])
        except ValueError:
            result.append(-1)  # 새로운 카테고리는 -1로 처리
    return np.array(result)

# --- 안전한 타겟 인코딩 ---
def safe_target_encoding(train_df, test_df, cat_cols, target_col, alpha=10, cv_folds=5):
    """안전한 타겟 인코딩 (교차 검증 사용)"""
    print(f"타겟 인코딩 적용: {cat_cols}")
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    target_encoders = {}
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
            
        print(f"  - {col} 인코딩 중...")
        train_encoded = np.zeros(len(train_df))
        
        # 교차 검증으로 안전한 타겟 인코딩
        for train_idx, val_idx in kf.split(train_df):
            train_part = train_df.iloc[train_idx]
            val_part = train_df.iloc[val_idx]
            
            # 베이지안 스무딩
            target_mean = train_part.groupby(col)[target_col].mean()
            global_mean = train_part[target_col].mean()
            count = train_part.groupby(col).size()
            
            smoothed = (target_mean * count + global_mean * alpha) / (count + alpha)
            train_encoded[val_idx] = val_part[col].map(smoothed).fillna(global_mean)
        
        train_df[f'{col}_target_enc'] = train_encoded
        
        # 테스트용: 전체 훈련 데이터로 계산
        target_mean = train_df.groupby(col)[target_col].mean()
        global_mean = train_df[target_col].mean()
        count = train_df.groupby(col).size()
        smoothed = (target_mean * count + global_mean * alpha) / (count + alpha)
        
        # 인코더 저장
        target_encoders[col] = {
            'smoothed': smoothed.to_dict(),
            'global_mean': global_mean
        }
        
        # 테스트 데이터에 적용
        if col in test_df.columns:
            test_df[f'{col}_target_enc'] = test_df[col].map(smoothed).fillna(global_mean)
        else:
            test_df[f'{col}_target_enc'] = global_mean
    
    return train_df, test_df, target_encoders

# --- 🌡️ 온도 기반 피처 생성 ---
def create_temperature_based_features(df):
    """온도 특성 기반 계절별 메뉴 피처 생성"""
    
    print("🌡️ 온도 기반 계절 메뉴 피처 생성 중...")
    
    # 메뉴별 온도 특성 정의 (도메인 지식 기반)
    menu_temperature_map = {
        # 뜨거운 음식 (겨울 선호)
        '뜨거운음식': {
            'keywords': ['국밥', '해장국', '찌개', '탕', '국수', '라면', '우동', '따뜻한', '온', '뜨거운', 
                        '김치찌개', '된장찌개', '부대찌개', '순두부', '삼계탕', '곰탕', '설렁탕', '갈비탕',
                        '미소라멘', '짜장면', '짬뽕', '불고기', '떡볶이', '어묵', '호떡', '붕어빵', '군고구마'],
            'season_preference': {'겨울': 1.5, '가을': 1.2, '봄': 0.8, '여름': 0.4}
        },
        
        # 시원한 음식 (여름 선호)  
        '시원한음식': {
            'keywords': ['냉면', '물냉면', '비빔냉면', '냉국수', '콩국수', '아이스크림', '빙수', '팥빙수', 
                        '냉커피', '아이스', '프라페', '스무디', '얼음', '시원한', '차가운', '냉', 
                        '샐러드', '과일', '요거트', '소르베', '젤라토'],
            'season_preference': {'여름': 1.5, '봄': 1.2, '가을': 0.8, '겨울': 0.4}
        },
        
        # 따뜻한 음료 (겨울 선호)
        '따뜻한음료': {
            'keywords': ['아메리카노', '라떼', '카푸치노', '마키아토', '모카', '핫초콜릿', '밀크티', 
                        '차', '녹차', '홍차', '허브차', '생강차', '유자차', '꿀차', '따뜻한'],
            'season_preference': {'겨울': 1.4, '가을': 1.3, '봄': 0.9, '여름': 0.5}
        },
        
        # 시원한 음료 (여름 선호)
        '시원한음료': {
            'keywords': ['콜라', '스프라이트', '사이다', '맥주', '소주', '하이볼', '칵테일', 
                        '에이드', '레모네이드', '생수', '탄산수', '주스', '쥬스'],
            'season_preference': {'여름': 1.4, '봄': 1.1, '가을': 0.9, '겨울': 0.6}
        }
    }
    
    # 각 메뉴의 온도 특성 분류
    for temp_type, info in menu_temperature_map.items():
        keywords = info['keywords']
        keyword_pattern = '|'.join(keywords)
        matches = df['메뉴명'].str.contains(keyword_pattern, case=False, na=False)
        df[f'메뉴_{temp_type}'] = matches.astype(int)
    
    print(f"  메뉴 온도 분류 결과:")
    for temp_type in menu_temperature_map.keys():
        count = df[f'메뉴_{temp_type}'].sum()
        print(f"    {temp_type}: {count:,}개 메뉴")
    
    # 월별 온도 가중치 (실제 한국 기후 반영)
    monthly_temp_weights = {
        1: -0.8, 2: -0.6, 3: 0.0, 4: 0.3, 5: 0.6, 6: 0.8,
        7: 1.0, 8: 1.0, 9: 0.5, 10: 0.2, 11: -0.2, 12: -0.6
    }
    
    df['월별_온도점수'] = df['월'].map(monthly_temp_weights).fillna(0)
    
    # 온도와 메뉴의 연속적 매칭 점수 (핵심 피처!)
    df['온도_메뉴_매칭점수'] = (
        df['월별_온도점수'] * df['메뉴_시원한음식'] * 1.0 +  # 더울수록 시원한 음식 선호
        df['월별_온도점수'] * df['메뉴_시원한음료'] * 0.8 +
        (1 - df['월별_온도점수']) * df['메뉴_뜨거운음식'] * 1.0 +  # 추울수록 뜨거운 음식 선호  
        (1 - df['월별_온도점수']) * df['메뉴_따뜻한음료'] * 0.8
    )
    
    # 계절별 온도 특성 매칭 피처 생성
    seasons = ['봄', '여름', '가을', '겨울']
    
    for temp_type, info in menu_temperature_map.items():
        season_pref = info['season_preference']
        
        for season in seasons:
            if season in df.columns:
                preference_score = season_pref.get(season, 1.0)
                df[f'{season}_{temp_type}_매칭'] = (
                    df[season] * df[f'메뉴_{temp_type}'] * preference_score
                )
    
    # 온도 대비 피처 (반대 계절 효과)
    df['겨울_시원한음식_대비'] = df['겨울'] * df['메뉴_시원한음식'] * 0.3
    df['여름_뜨거운음식_대비'] = df['여름'] * df['메뉴_뜨거운음식'] * 0.4
    df['여름_따뜻한음료_대비'] = df['여름'] * df['메뉴_따뜻한음료'] * 0.5
    df['겨울_시원한음료_대비'] = df['겨울'] * df['메뉴_시원한음료'] * 0.6
    
    # 영업장별 온도 특성 (각 영업장의 특색 반영)
    stores = ['포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아']
    
    for store in stores:
        if f'영업장_{store}' in df.columns:
            if store == '화담숲주막':  # 전통 주막 → 따뜻한 음식 특화
                df[f'{store}_뜨거운음식_특화'] = df[f'영업장_{store}'] * df['메뉴_뜨거운음식'] * 1.3
                df[f'{store}_겨울_시너지'] = df[f'영업장_{store}'] * df['겨울'] * 1.2
                
            elif store == '포레스트릿':  # 관광지 → 계절성 강함
                df[f'{store}_계절매칭_강화'] = (
                    df[f'영업장_{store}'] * (
                        df['여름_시원한음식_매칭'] + df['겨울_뜨거운음식_매칭']
                    )
                )
                
            elif store == '카페테리아':  # 카페 → 음료 특화
                df[f'{store}_음료_계절매칭'] = (
                    df[f'영업장_{store}'] * (
                        df['여름_시원한음료_매칭'] + df['겨울_따뜻한음료_매칭']
                    )
                )
    
    # 극단적 계절 상황 피처
    df['혹한기_뜨거운음식'] = df['메뉴_뜨거운음식'] * (df['월'].isin([12, 1, 2])).astype(int) * 1.5
    df['혹서기_시원한음식'] = df['메뉴_시원한음식'] * (df['월'].isin([7, 8])).astype(int) * 1.5
    
    # 특별한 날씨 이벤트 (한국 기후 특성)
    df['장마철_실내음식'] = (df['월'].isin([6, 7])).astype(int) * df['메뉴_뜨거운음식'] * 1.2
    df['폭염_시원음식'] = (df['월'].isin([7, 8])).astype(int) * df['메뉴_시원한음식'] * 1.5
    df['한파_따뜻음식'] = (df['월'].isin([12, 1, 2])).astype(int) * df['메뉴_뜨거운음식'] * 1.3
    
    print(f"  온도 기반 피처 생성 완료!")
    
    return df

# --- 고급 시간 피처 생성 ---
def create_advanced_time_features(df):
    """고급 시간 피처 생성"""
    df['영업일자_dt'] = pd.to_datetime(df['영업일자'])
    
    # 기본 시간 피처
    df['년'] = df['영업일자_dt'].dt.year
    df['월'] = df['영업일자_dt'].dt.month
    df['일'] = df['영업일자_dt'].dt.day
    df['요일'] = df['영업일자_dt'].dt.dayofweek
    df['주차'] = df['영업일자_dt'].dt.isocalendar().week
    df['분기'] = df['영업일자_dt'].dt.quarter
    df['년중_몇째날'] = df['영업일자_dt'].dt.dayofyear
    
    # 삼각함수를 이용한 주기성 표현
    df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
    df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
    df['요일_sin'] = np.sin(2 * np.pi * df['요일'] / 7)
    df['요일_cos'] = np.cos(2 * np.pi * df['요일'] / 7)
    df['일_sin'] = np.sin(2 * np.pi * df['일'] / 31)
    df['일_cos'] = np.cos(2 * np.pi * df['일'] / 31)
    
    # 계절성
    df['봄'] = df['월'].isin([3, 4, 5]).astype(int)
    df['여름'] = df['월'].isin([6, 7, 8]).astype(int)
    df['가을'] = df['월'].isin([9, 10, 11]).astype(int)
    df['겨울'] = df['월'].isin([12, 1, 2]).astype(int)
    
    # 특별한 월
    df['1월'] = (df['월'] == 1).astype(int)
    df['3월'] = (df['월'] == 3).astype(int)
    df['5월'] = (df['월'] == 5).astype(int)
    df['8월'] = (df['월'] == 8).astype(int)
    df['12월'] = (df['월'] == 12).astype(int)
    
    # 월의 특성
    df['월초'] = (df['일'] <= 5).astype(int)
    df['월중'] = ((df['일'] > 5) & (df['일'] <= 25)).astype(int)
    df['월말'] = (df['일'] > 25).astype(int)
    df['급여일근처'] = ((df['일'] >= 23) & (df['일'] <= 28)).astype(int)
    
    # 주말/공휴일
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['공휴일여부'] = df['영업일자_dt'].dt.strftime('%Y-%m-%d').isin(holidays_2023_2024).astype(int)
    df['금요일'] = (df['요일'] == 4).astype(int)
    df['일요일'] = (df['요일'] == 6).astype(int)
    
    return df

# --- 상호작용 피처 생성 ---
def create_interaction_features(df):
    """상호작용 피처 생성 (온도 기반 포함)"""
    # 기존 상호작용 피처들
    seasons = ['봄', '여름', '가을', '겨울']
    stores = ['포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아']
    
    for season in seasons:
        for store in stores:
            if f'{season}' in df.columns and f'영업장_{store}' in df.columns:
                df[f'{store}_{season}'] = df[f'{season}'] * df[f'영업장_{store}']
    
    # 메뉴 카테고리 × 요일
    menu_categories = ['음료류', '주류류', '분식류', '한식류', '양식류', '디저트류']
    for category in menu_categories:
        if category in df.columns:
            df[f'{category}_주말'] = df[category] * df['주말여부']
            df[f'{category}_평일'] = df[category] * (1 - df['주말여부'])
    
    # 영업장 × 주말
    for store in stores:
        if f'영업장_{store}' in df.columns:
            df[f'{store}_주말'] = df[f'영업장_{store}'] * df['주말여부']
    
    # 월 × 주말
    special_months = [1, 3, 5, 8, 12]
    for month in special_months:
        if f'{month}월' in df.columns:
            df[f'{month}월_주말'] = df[f'{month}월'] * df['주말여부']
    
    return df

# --- 이상치 처리 ---
def handle_outliers_by_group(df, target_col='매출수량'):
    """그룹별 이상치 처리"""
    if target_col not in df.columns:
        return df
    
    print("이상치 처리 중...")
    original_mean = df[target_col].mean()
    
    # 영업장별 이상치 처리
    df[f'{target_col}_clipped'] = df.groupby('영업장명')[target_col].transform(
        lambda x: x.clip(lower=x.quantile(0.02), upper=x.quantile(0.98))
    )
    
    # 처리 결과 확인
    processed_mean = df[f'{target_col}_clipped'].mean()
    print(f"  이상치 처리 전 평균: {original_mean:.2f}")
    print(f"  이상치 처리 후 평균: {processed_mean:.2f}")
    
    # 원본 대신 처리된 값 사용
    df[target_col] = df[f'{target_col}_clipped']
    df = df.drop(f'{target_col}_clipped', axis=1)
    
    return df

# --- 통합 피처 생성 ---
def create_ultimate_features(df, is_train=True, encoders=None, target_encoders=None, train_df_for_target=None):
    """모든 고급 피처를 통합한 최종 피처 생성"""
    df = df.copy()
    
    # 1. 고급 시간 피처
    df = create_advanced_time_features(df)
    
    # 2. 영업장/메뉴 분리
    df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
    df['메뉴명'] = df['영업장명_메뉴명'].str.split('_', n=1).str[1].fillna('')
    
    # 3. 영업장별 원핫 인코딩
    unique_stores = ['포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아']
    for store in unique_stores:
        df[f'영업장_{store}'] = (df['영업장명'] == store).astype(int)
    
    # 4. 메뉴 카테고리
    df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|음료', na=False).astype(int)
    df['주류류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|와인|칵테일|하이볼', na=False).astype(int)
    df['분식류'] = df['메뉴명'].str.contains('떡볶이|튀김|핫도그|어묵|꼬치', na=False).astype(int)
    df['한식류'] = df['메뉴명'].str.contains('국밥|불고기|김치|된장|비빔밥|한식', na=False).astype(int)
    df['양식류'] = df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|양식', na=False).astype(int)
    df['디저트류'] = df['메뉴명'].str.contains('케이크|빵|과자|디저트|아이스크림', na=False).astype(int)
    df['세트메뉴'] = df['메뉴명'].str.contains('세트|패키지|콤보', na=False).astype(int)
    
    # 5. 🌡️ 온도 기반 피처 생성 (핵심!)
    df = create_temperature_based_features(df)
    
    # 6. 상호작용 피처
    df = create_interaction_features(df)
    
    # 7. 카테고리형 변수 인코딩
    categorical_features = ['영업장명', '메뉴명', '영업장명_메뉴명']
    
    if is_train:
        # 훈련 시: 새로운 인코더 생성
        if encoders is None:
            encoders = {}
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        # 이상치 처리
        df = handle_outliers_by_group(df, '매출수량')
        
        return df, encoders
    else:
        # 테스트 시: 기존 인코더 사용
        for col in categorical_features:
            if col in df.columns and encoders and col in encoders:
                df[f'{col}_encoded'] = safe_label_encode(encoders[col], df[col])
            else:
                df[f'{col}_encoded'] = -1
        
        # 타겟 인코딩 적용 (테스트)
        if target_encoders and train_df_for_target is not None:
            target_cols = ['영업장명', '메뉴명', '영업장명_메뉴명']
            _, df, _ = safe_target_encoding(train_df_for_target, df, target_cols, '매출수량')
        
        return df

# --- 앙상블 모델 학습 ---
def train_ensemble_model_ultimate(train_df):
    """온도 기반 피처를 포함한 고급 앙상블 모델 학습"""
    print("\n" + "="*80)
    print("🚀 온도 기반 피처 포함 고급 앙상블 모델 학습")
    print("="*80)
    
    # 1. 피처 생성
    print("1️⃣ 고급 피처 생성 중...")
    train_df_processed, encoders = create_ultimate_features(train_df, is_train=True)
    
    # 2. 피처 선택
    print("2️⃣ 피처 선택 중...")
    feature_columns = [col for col in train_df_processed.columns 
                      if col not in ['영업일자', '영업장명_메뉴명', '매출수량', '영업일자_dt']]
    
    # 수치형 피처와 카테고리형 피처 분리
    numeric_features = [col for col in feature_columns if col not in ['영업장명', '메뉴명']]
    categorical_features = [col for col in feature_columns if col in ['영업장명', '메뉴명']]
    
    # 3. 모델 학습
    print("3️⃣ 앙상블 모델 학습 중...")
    models = {}
    
    # CatBoost
    print("  - CatBoost 학습 중...")
    catboost_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100
    )
    catboost_model.fit(
        train_df_processed[numeric_features],
        train_df_processed['매출수량'],
        cat_features=[i for i, col in enumerate(numeric_features) if col in categorical_features]
    )
    models['catboost'] = catboost_model
    
    # LightGBM
    print("  - LightGBM 학습 중...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(train_df_processed[numeric_features], train_df_processed['매출수량'])
    models['lightgbm'] = lgb_model
    
    # XGBoost
    print("  - XGBoost 학습 중...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(train_df_processed[numeric_features], train_df_processed['매출수량'])
    models['xgboost'] = xgb_model
    
    print("✅ 앙상블 모델 학습 완료!")
    
    return models, encoders, numeric_features, train_df_processed

# --- 고급 피처 중요도 분석 ---
def analyze_feature_importance_ultimate(models, feature_names, train_df_processed, target_col):
    """온도 피처 포함 포괄적인 피처 중요도 분석"""
    print("\n" + "="*80)
    print("🔍 온도 기반 피처 포함 포괄적인 피처 중요도 분석")
    print("="*80)
    
    importance_results = {}
    
    # 1. 각 모델별 피처 중요도
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_results[model_name] = model.feature_importances_
    
    # 2. 통계적 중요도 (F-score)
    X = train_df_processed[feature_names]
    y = train_df_processed[target_col]
    
    # 결측치 처리
    for col in feature_names:
        if col.endswith('_encoded'):
            X[col] = X[col].fillna(-1)
        else:
            X[col] = X[col].fillna(0)
    
    f_scores, _ = f_regression(X, y)
    importance_results['f_score'] = f_scores
    
    # 3. 상관관계 기반 중요도
    correlations = []
    for col in feature_names:
        if col in X.columns:
            corr = abs(X[col].corr(y))
            correlations.append(corr)
        else:
            correlations.append(0)
    importance_results['correlation'] = np.array(correlations)
    
    # 4. 종합 중요도 계산 (가중평균)
    weights = {
        'catboost': 0.35,
        'lightgbm': 0.3,
        'xgboost': 0.25,
        'f_score': 0.1
    }
    
    # 정규화
    normalized_importance = {}
    for method, scores in importance_results.items():
        if method != 'correlation':
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            normalized_importance[method] = scores_norm
    
    # 종합 점수 계산
    combined_score = np.zeros(len(feature_names))
    for method, weight in weights.items():
        if method in normalized_importance:
            combined_score += weight * normalized_importance[method]
    
    # 피처 중요도 DataFrame 생성
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'combined_score': combined_score,
        'catboost_importance': importance_results.get('catboost', np.zeros(len(feature_names))),
        'lightgbm_importance': importance_results.get('lightgbm', np.zeros(len(feature_names))),
        'xgboost_importance': importance_results.get('xgboost', np.zeros(len(feature_names))),
        'f_score': importance_results.get('f_score', np.zeros(len(feature_names))),
        'correlation': importance_results.get('correlation', np.zeros(len(feature_names)))
    }).sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    # 피처 카테고리 분류 (온도 기반 추가)
    def categorize_feature_ultimate(feature_name):
        if 'target_enc' in feature_name:
            return '🎯 타겟 인코딩'
        elif 'encoded' in feature_name:
            return '🏷️ 카테고리 인코딩'
        elif any(temp in feature_name for temp in ['뜨거운', '시원한', '따뜻한', '온도', '매칭', '대비', '특화', '시너지', '혹한', '혹서', '장마', '폭염', '한파']):
            return '🌡️ 온도 기반 피처'
        elif any(time in feature_name for time in ['년', '월', '일', '요일', '주차', '분기', '봄', '여름', '가을', '겨울']):
            return '⏰ 시간 피처'
        elif any(store in feature_name for store in ['포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아']):
            return '🏪 영업장 피처'
        elif any(cat in feature_name for cat in ['음료류', '주류류', '분식류', '한식류', '양식류', '디저트류', '세트메뉴']):
            return '🍽️ 메뉴 카테고리'
        elif 'sin' in feature_name or 'cos' in feature_name:
            return '📊 주기성 피처'
        else:
            return '📈 기타 피처'
    
    feature_importance_df['category'] = feature_importance_df['feature'].apply(categorize_feature_ultimate)
    
    # 카테고리별 중요도 분석
    print("\n📊 카테고리별 피처 중요도:")
    category_importance = feature_importance_df.groupby('category')['combined_score'].mean().sort_values(ascending=False)
    for category, score in category_importance.items():
        print(f"  {category}: {score:.4f}")
    
    # 상위 피처 출력
    print(f"\n🏆 상위 20개 피처:")
    top_features = feature_importance_df.head(20)
    for idx, row in top_features.iterrows():
        print(f"  {idx+1:2d}. {row['feature']:<30} (점수: {row['combined_score']:.4f}, 카테고리: {row['category']})")
    
    return feature_importance_df

# --- 앙상블 예측 ---
def predict_ensemble_ultimate(test_df, models, encoders, feature_columns):
    """온도 기반 피처를 포함한 앙상블 예측"""
    # 피처 생성
    test_df_processed = create_ultimate_features(test_df, is_train=False, encoders=encoders)
    
    # 예측
    predictions = {}
    for name, model in models.items():
        pred = model.predict(test_df_processed[feature_columns])
        predictions[name] = pred
    
    # 앙상블 예측 (가중 평균)
    weights = {'catboost': 0.4, 'lightgbm': 0.35, 'xgboost': 0.25}
    ensemble_pred = np.zeros(len(test_df))
    
    for name, pred in predictions.items():
        ensemble_pred += weights[name] * pred
    
    return ensemble_pred

# --- 제출 파일 변환 ---
def convert_to_submission_format_ultimate(pred_df, sample_submission):
    """예측 결과를 제출 형식으로 변환 (정수 변환 포함)"""
    def convert_to_integer(value):
        if value < 0:
            return 0
        return max(0, round(value))
    
    # 예측 결과를 딕셔너리로 변환
    pred_dict = {}
    for _, row in pred_df.iterrows():
        date = row['영업일자']
        menu = row['영업장명_메뉴명']
        value = convert_to_integer(row['매출수량'])
        pred_dict[(date, menu)] = value
    
    # 샘플 제출 파일 형식에 맞춰 결과 생성
    final_df = sample_submission.copy()
    
    for idx, row in final_df.iterrows():
        date = row['영업일자']
        for col in final_df.columns[1:]:  # 첫 번째 컬럼(영업일자) 제외
            value = pred_dict.get((date, col), 0)
            final_df.at[idx, col] = value
    
    return final_df

# --- 메인 실행 ---
if __name__ == "__main__":
    print("🌡️ 온도 기반 피처를 포함한 고급 앙상블 모델")
    print("="*80)
    
    # 1. 데이터 로드
    print("1️⃣ 데이터 로드 중...")
    train = pd.read_csv('./data/train/train.csv')
    print(f"   훈련 데이터: {train.shape}")
    
    # 2. 앙상블 모델 학습
    models, encoders, feature_columns, train_df_processed = train_ensemble_model_ultimate(train)
    
    # 3. 피처 중요도 분석
    print("\n4️⃣ 피처 중요도 분석 중...")
    feature_importance_df = analyze_feature_importance_ultimate(
        models, feature_columns, train_df_processed, '매출수량'
    )
    
    # 4. 테스트 데이터 예측
    print("\n5️⃣ 테스트 데이터 예측 중...")
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    all_preds = []
    
    for test_file in sorted(glob.glob('./data/test/TEST_*.csv')):
        print(f"   처리 중: {test_file}")
        test_df = pd.read_csv(test_file)
        
        # 앙상블 예측
        predictions = predict_ensemble_ultimate(test_df, models, encoders, feature_columns)
        
        # 날짜 형식 변환 (TEST_XX+N일 형식으로)
        filename = os.path.basename(test_file)
        test_prefix = filename.replace('.csv', '')
        base_date = pd.to_datetime(test_df['영업일자'].iloc[0])
        
        converted_dates = test_df['영업일자'].apply(
            lambda x: f"{test_prefix}+{(pd.to_datetime(x) - base_date).days + 1}일"
        )
        
        # 예측 결과 DataFrame 생성
        pred_df = pd.DataFrame({
            '영업일자': converted_dates,
            '영업장명_메뉴명': test_df['영업장명_메뉴명'],
            '매출수량': predictions
        })
        all_preds.append(pred_df)
    
    # 모든 예측 결과 합치기
    full_pred_df = pd.concat(all_preds, ignore_index=True)
    
    # 5. 제출 파일 생성
    print("\n6️⃣ 제출 파일 생성 중...")
    submission = convert_to_submission_format_ultimate(full_pred_df, sample_submission)
    
    # 6. 결과 저장
    output_file = 'temperature_ensemble_submission.csv'
    submission.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 예측 완료! 결과가 '{output_file}'에 저장되었습니다.")
    
    # 7. 결과 요약
    print(f"\n📊 결과 요약:")
    print(f"   - 총 예측 행 수: {len(submission)}")
    print(f"   - 총 메뉴 수: {len(submission.columns) - 1}")
    print(f"   - 평균 예측값: {submission.iloc[:, 1:].values.mean():.2f}")
    print(f"   - 최대 예측값: {submission.iloc[:, 1:].values.max():.2f}")
    print(f"   - 최소 예측값: {submission.iloc[:, 1:].values.min():.2f}")
    
    print("\n🎉 온도 기반 피처를 포함한 고급 앙상블 모델 실행 완료!")