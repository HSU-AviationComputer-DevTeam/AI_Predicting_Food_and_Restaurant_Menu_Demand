import os
import random
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import VotingRegressor
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
    
    # 급여일 효과
    df['급여일근처'] = ((df['일'] >= 23) & (df['일'] <= 28)).astype(int)
    
    # 주말/공휴일
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['공휴일여부'] = df['영업일자_dt'].dt.strftime('%Y-%m-%d').isin(holidays_2023_2024).astype(int)
    df['금요일'] = (df['요일'] == 4).astype(int)
    df['일요일'] = (df['요일'] == 6).astype(int)
    
    return df

# --- 상호작용 피처 생성 ---
def create_interaction_features(df):
    """상호작용 피처 생성"""
    # 영업장 × 계절
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

# --- 개선된 피처 생성 ---
def create_improved_features(df, is_train=True, encoders=None, target_encoders=None, train_df_for_target=None):
    """개선된 피처 생성"""
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
    
    # 5. 상호작용 피처
    df = create_interaction_features(df)
    
    # 6. 카테고리형 변수 인코딩
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

# --- 고급 피처 중요도 분석 ---
def analyze_feature_importance_comprehensive(models, feature_names, train_df, target_col):
    """포괄적인 피처 중요도 분석"""
    print("\n" + "="*60)
    print("🔍 포괄적인 피처 중요도 분석")
    print("="*60)
    
    importance_results = {}
    
    # 1. 각 모델별 피처 중요도
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_results[model_name] = model.feature_importances_
    
    # 2. 통계적 중요도 (F-score)
    X = train_df[feature_names]
    y = train_df[target_col]
    
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
            # min-max 정규화
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
    
    # 피처 카테고리 분류
    def categorize_feature(feature_name):
        if 'target_enc' in feature_name:
            return '🎯 타겟 인코딩'
        elif 'encoded' in feature_name:
            return '🏷️ 카테고리 인코딩'
        elif any(store in feature_name for store in ['포레스트릿', '카페테리아', '화담숲주막', '담하', '미라시아']):
            if '_' in feature_name and feature_name.count('_') > 1:
                return '🔗 영업장 상호작용'
            else:
                return '🏪 영업장'
        elif any(menu in feature_name for menu in ['음료류', '주류류', '분식류', '한식류', '양식류', '디저트류', '세트메뉴']):
            if '_' in feature_name and feature_name.count('_') > 0:
                return '🔗 메뉴 상호작용'
            else:
                return '🍽️ 메뉴 카테고리'
        elif feature_name in ['년', '월', '일', '요일', '주차', '분기', '년중_몇째날']:
            return '📅 기본 시간'
        elif 'sin' in feature_name or 'cos' in feature_name:
            return '〰️ 삼각함수'
        elif feature_name in ['봄', '여름', '가을', '겨울']:
            return '🌱 계절성'
        elif feature_name in ['1월', '3월', '5월', '8월', '12월']:
            return '📆 특별한 월'
        elif feature_name in ['월초', '월중', '월말', '급여일근처']:
            return '📊 월 특성'
        elif feature_name in ['주말여부', '공휴일여부', '금요일', '일요일']:
            return '🎉 주말/공휴일'
        else:
            return '❓ 기타'
    
    feature_importance_df['category'] = feature_importance_df['feature'].apply(categorize_feature)
    
    # 결과 출력
    print(f"\n📊 전체 피처 중요도 순위 (Top 30)")
    print("-" * 100)
    print(f"{'순위':>3} {'피처명':<35} {'카테고리':<20} {'종합점수':>8} {'CatBoost':>8} {'LightGBM':>8} {'XGBoost':>8}")
    print("-" * 100)
    
    for idx, row in feature_importance_df.head(30).iterrows():
        print(f"{idx+1:>3} {row['feature']:<35} {row['category']:<20} "
              f"{row['combined_score']:>8.4f} {row['catboost_importance']:>8.4f} "
              f"{row['lightgbm_importance']:>8.4f} {row['xgboost_importance']:>8.4f}")
    
    # 카테고리별 중요도 요약
    print(f"\n📈 카테고리별 중요도 요약")
    print("-" * 50)
    category_importance = feature_importance_df.groupby('category').agg({
        'combined_score': ['mean', 'sum', 'count']
    }).round(4)
    category_importance.columns = ['평균_중요도', '총합_중요도', '피처_수']
    category_importance = category_importance.sort_values('총합_중요도', ascending=False)
    
    for category, row in category_importance.iterrows():
        print(f"{category:<25} 평균:{row['평균_중요도']:>7.4f} 총합:{row['총합_중요도']:>7.4f} 개수:{row['피처_수']:>3.0f}")
    
    # 추천 피처 선택
    print(f"\n🎯 추천 피처 선택 전략")
    print("-" * 50)
    
    # Top K 피처들 선택 전략
    strategies = {
        "보수적 (Top 30)": 30,
        "균형적 (Top 40)": 40, 
        "공격적 (Top 50)": 50
    }
    
    for strategy_name, k in strategies.items():
        selected_features = feature_importance_df.head(k)['feature'].tolist()
        category_dist = selected_features = feature_importance_df.head(k)['category'].value_counts()
        
        print(f"\n{strategy_name}:")
        print(f"  선택 피처 수: {k}개")
        print(f"  주요 카테고리 구성:")
        for cat, count in category_dist.head(5).items():
            print(f"    - {cat}: {count}개")
    
    return feature_importance_df

# --- 피처 선택 모델 학습 ---
def train_feature_selected_model(train_df, target_col='매출수량', n_features=40):
    """피처 선택 기반 모델 학습"""
    
    print("=== 피처 선택 기반 모델 학습 ===")
    print(f"목표: 90개 → {n_features}개 핵심 피처 선택")
    
    # 피처 생성
    print("피처 생성 중...")
    train_df, encoders = create_improved_features(train_df, is_train=True)
    
    # 타겟 인코딩 적용
    target_cols = ['영업장명', '메뉴명', '영업장명_메뉴명']
    dummy_test = train_df.head(1).copy()
    train_df, _, target_encoders = safe_target_encoding(train_df, dummy_test, target_cols, target_col)
    
    # 전체 피처 목록
    feature_columns = [
        # 기본 시간 피처
        '년', '월', '일', '요일', '주차', '분기', '년중_몇째날',
        # 삼각함수 피처
        '월_sin', '월_cos', '요일_sin', '요일_cos', '일_sin', '일_cos',
        # 계절성
        '봄', '여름', '가을', '겨울',
        # 특별한 월
        '1월', '3월', '5월', '8월', '12월',
        # 월의 특성
        '월초', '월중', '월말', '급여일근처',
        # 주말/공휴일
        '주말여부', '공휴일여부', '금요일', '일요일',
        # 영업장
        '영업장_포레스트릿', '영업장_카페테리아', '영업장_화담숲주막', '영업장_담하', '영업장_미라시아',
        # 메뉴 카테고리
        '음료류', '주류류', '분식류', '한식류', '양식류', '디저트류', '세트메뉴',
        # 카테고리 인코딩
        '영업장명_encoded', '메뉴명_encoded', '영업장명_메뉴명_encoded',
        # 타겟 인코딩
        '영업장명_target_enc', '메뉴명_target_enc', '영업장명_메뉴명_target_enc'
    ]
    
    # 상호작용 피처들 추가
    interaction_features = [col for col in train_df.columns if '_주말' in col or '_봄' in col or '_여름' in col or '_가을' in col or '_겨울' in col or '_평일' in col]
    feature_columns.extend(interaction_features)
    
    # 존재하는 피처만 선택
    available_features = [col for col in feature_columns if col in train_df.columns]
    print(f"전체 사용 가능한 피처 수: {len(available_features)}")
    
    # 결측치 처리
    for col in available_features:
        if col.endswith('_encoded'):
            train_df[col] = train_df[col].fillna(-1)
        else:
            train_df[col] = train_df[col].fillna(0)
    
    # 개별 모델들 생성 (피처 중요도 분석용)
    individual_models = {
        'catboost': CatBoostRegressor(
            iterations=300, learning_rate=0.08, depth=7, l2_leaf_reg=2,
            random_seed=42, verbose=False
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=7, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
    }
    
    # 전체 데이터로 모델 학습 (피처 중요도 분석용)
    X_full = train_df[available_features]
    y_full = train_df[target_col]
    
    print("\n피처 중요도 분석을 위한 모델 학습...")
    trained_models = {}
    cat_features = [i for i, col in enumerate(available_features) if col.endswith('_encoded')]
    
    for name, model in individual_models.items():
        if name == 'catboost':
            model.fit(X_full, y_full, cat_features=cat_features)
        else:
            model.fit(X_full, y_full)
        trained_models[name] = model
    
    # 포괄적인 피처 중요도 분석
    feature_importance_df = analyze_feature_importance_comprehensive(
        trained_models, available_features, train_df, target_col
    )
    
    # Top K 피처 선택
    selected_features = feature_importance_df.head(n_features)['feature'].tolist()
    print(f"\n✅ 선택된 {len(selected_features)}개 핵심 피처:")
    
    # 선택된 피처를 카테고리별로 출력
    selected_df = feature_importance_df.head(n_features)
    for category in selected_df['category'].unique():
        category_features = selected_df[selected_df['category'] == category]['feature'].tolist()
        print(f"  {category}: {len(category_features)}개")
        for feature in category_features[:3]:  # 처음 3개만 표시
            rank = selected_df[selected_df['feature'] == feature].index[0] + 1
            score = selected_df[selected_df['feature'] == feature]['combined_score'].iloc[0]
            print(f"    {rank:2d}. {feature:<30} (점수: {score:.4f})")
        if len(category_features) > 3:
            print(f"    ... 외 {len(category_features)-3}개")
    
    # 선택된 피처로 교차 검증
    print(f"\n=== 피처 선택 모델 교차 검증 ===")
    tscv = TimeSeriesSplit(n_splits=5)
    
    X_selected = train_df[selected_features]
    y = train_df[target_col]
    
    # 결측치 재처리
    for col in selected_features:
        if col.endswith('_encoded'):
            X_selected[col] = X_selected[col].fillna(-1)
        else:
            X_selected[col] = X_selected[col].fillna(0)
    
    model_scores = {}
    ensemble_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
        print(f"\nFold {fold + 1} 평가 중...")
        
        X_train_fold = X_selected.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X_selected.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        fold_predictions = {}
        
        # 선택된 피처에서 카테고리 피처 재정의
        selected_cat_features = [i for i, col in enumerate(selected_features) if col.endswith('_encoded')]
        
        for name, model in individual_models.items():
            if name == 'catboost':
                model.fit(X_train_fold, y_train_fold, cat_features=selected_cat_features)
            else:
                model.fit(X_train_fold, y_train_fold)
            
            pred = model.predict(X_val_fold)
            pred = np.maximum(pred, 0)
            
            rmse = np.sqrt(np.mean((y_val_fold - pred) ** 2))
            fold_predictions[name] = pred
            
            if name not in model_scores:
                model_scores[name] = []
            model_scores[name].append(rmse)
            
            print(f"  {name:>10}: {rmse:.4f}")
        
        # 앙상블 예측
        ensemble_pred = (
            0.4 * fold_predictions['catboost'] +
            0.35 * fold_predictions['lightgbm'] +
            0.25 * fold_predictions['xgboost']
        )
        
        ensemble_rmse = np.sqrt(np.mean((y_val_fold - ensemble_pred) ** 2))
        ensemble_predictions.append(ensemble_rmse)
        
        print(f"  {'앙상블':>10}: {ensemble_rmse:.4f}")
    
    # 성능 요약
    print(f"\n=== 피처 선택 모델 성능 요약 ===")
    for name, scores in model_scores.items():
        print(f"{name:>12}: 평균 {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    
    ensemble_mean = np.mean(ensemble_predictions)
    ensemble_std = np.std(ensemble_predictions)
    print(f"{'앙상블':>12}: 평균 {ensemble_mean:.4f} (±{ensemble_std:.4f})")
    print(f"{'안정성 지수':>12}: {ensemble_std/ensemble_mean:.3f}")
    
    # 이전 결과와 비교
    previous_score = 22.71  # 이전 앙상블 모델 결과
    improvement = previous_score - ensemble_mean
    improvement_pct = (improvement / previous_score) * 100
    
    print(f"\n🎉 피처 선택 효과:")
    print(f"  이전 모델 (90개 피처): {previous_score:.2f}")
    print(f"  선택 모델 ({n_features}개 피처): {ensemble_mean:.2f}")
    print(f"  성능 변화: {improvement:+.2f} ({improvement_pct:+.1f}%)")
    print(f"  피처 감소: 90개 → {n_features}개 ({(90-n_features)/90*100:.0f}% 감소)")
    
    # 최종 모델 학습
    print(f"\n최종 피처 선택 모델 학습 중...")
    final_models = {}
    
    for name, model in individual_models.items():
        print(f"  {name} 최종 학습 중...")
        if name == 'catboost':
            model.fit(X_selected, y, cat_features=selected_cat_features)
        else:
            model.fit(X_selected, y)
        final_models[name] = model
    
    return final_models, encoders, selected_features, target_encoders, train_df, feature_importance_df

# --- 피처 선택 예측 함수 ---
def predict_feature_selected(test_df, models, encoders, feature_columns, target_encoders, train_df_for_target):
    """피처 선택 모델로 예측"""
    
    # 피처 생성
    test_df = create_improved_features(test_df, is_train=False, encoders=encoders, 
                                     target_encoders=target_encoders, train_df_for_target=train_df_for_target)
    
    # 결측치 처리
    for col in feature_columns:
        if col in test_df.columns:
            if col.endswith('_encoded'):
                test_df[col] = test_df[col].fillna(-1)
            else:
                test_df[col] = test_df[col].fillna(0)
        else:
            test_df[col] = 0
    
    # 각 모델별 예측
    predictions = {}
    for name, model in models.items():
        pred = model.predict(test_df[feature_columns])
        pred = np.maximum(pred, 0)
        predictions[name] = pred
    
    # 앙상블 예측
    ensemble_pred = (
        0.4 * predictions['catboost'] +
        0.35 * predictions['lightgbm'] +
        0.25 * predictions['xgboost']
    )
    
    return ensemble_pred

# --- 제출 파일 변환 ---
def convert_to_submission_format(pred_df, sample_submission):
    """예측 결과를 제출 형식으로 변환"""
    def convert_to_integer(value):
        return max(0, int(round(value)))
    
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
    print("=== 🎯 피처 선택 최적화 모델 ===")
    print("목표:")
    print("  📊 피처 수: 90개 → 40개 (핵심만 선택)")
    print("  📈 성능: 오버피팅 방지로 안정성 향상")
    print("  🔍 해석력: 중요 피처 명확한 랭킹 제공")
    print("  🚀 효율성: 학습/예측 속도 향상")
    print()

    # 1. 데이터 로드
    print("데이터 로드 중...")
    train = pd.read_csv('./data/train/train.csv')
    print(f"훈련 데이터: {train.shape}")

    # 2. 피처 선택 모델 학습
    print("\n피처 선택 모델 학습 중...")
    models, encoders, selected_features, target_encoders, processed_train, feature_importance_df = train_feature_selected_model(train, n_features=40)

    # 3. 테스트 데이터 예측
    print("\n테스트 데이터 예측 중...")
    all_preds = []
    
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    for i, path in enumerate(test_files):
        print(f"처리 중 ({i+1}/{len(test_files)}): {os.path.basename(path)}")
        
        test_df = pd.read_csv(path)
        preds = predict_feature_selected(test_df, models, encoders, selected_features, target_encoders, processed_train)
        
        # 날짜 변환
        filename = os.path.basename(path)
        test_prefix = filename.replace('.csv', '')
        
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
    
    # 4. 결합 및 제출 파일 생성
    full_pred_df = pd.concat(all_preds, ignore_index=True)
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    submission = convert_to_submission_format(full_pred_df, sample_submission)

    # 5. 결과 저장
    submission.to_csv('feature_selected_submission.csv', index=False, encoding='utf-8-sig')
    
    # 6. 피처 중요도 상세 분석 저장
    feature_importance_df.to_csv('feature_importance_analysis.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== 🎉 피처 선택 모델 완료 ===")
    print("✅ 결과가 'feature_selected_submission.csv'에 저장되었습니다.")
    print("✅ 피처 중요도 분석이 'feature_importance_analysis.csv'에 저장되었습니다.")
    
    print(f"\n📊 주요 성과:")
    print("🔬 정밀한 피처 중요도 분석 (4가지 방법 종합)")
    print("📉 피처 수 56% 감소 (90개 → 40개)")  
    print("⚡ 학습/예측 속도 향상")
    print("🎯 핵심 피처만 사용으로 해석력 향상")
    print("📈 오버피팅 방지로 안정성 개선 기대")
    
    print(f"\n🏆 모델 진화 과정:")
    print("1단계 단순모델:     RMSE 32.77 (33개 피처)")
    print("2단계 타겟인코딩:   RMSE 22.76 (90개 피처, 30.6% 개선)")
    print("3단계 앙상블모델:   RMSE 22.71 (90개 피처, 안정성 향상)") 
    print("4단계 피처선택:     RMSE ??.?? (40개 피처, 최적화 완료)")
    
    print(f"\n💡 피처 중요도 인사이트:")
    print("📈 가장 중요한 피처 Top 5:")
    for i, row in feature_importance_df.head(5).iterrows():
        print(f"  {i+1}. {row['feature']:<30} ({row['category']}) - {row['combined_score']:.4f}")
    
    print(f"\n🔍 상세 분석:")
    print("- 'feature_importance_analysis.csv' 파일에서 전체 피처 순위 확인 가능")
    print("- 4가지 중요도 측정 방법의 종합 점수 제공")
    print("- 카테고리별 기여도 분석 포함")
    
    print(f"\n🚀 다음 단계 개선 방향:")
    print("💡 하이퍼파라미터 베이지안 최적화")
    print("💡 스태킹 앙상블 (2단계)")
    print("💡 시간 가중 타겟 인코딩") 
    print("💡 고급 정규화 기법")