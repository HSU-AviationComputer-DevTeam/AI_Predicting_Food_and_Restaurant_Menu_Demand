import os
import random
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from catboost import CatBoostRegressor
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
    df['금요일'] = (df['요일'] == 4).astype(int)  # 불금
    df['일요일'] = (df['요일'] == 6).astype(int)  # 일요일
    
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
    
    # 월 × 주말 (특정 월의 주말 효과)
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

# --- 개선된 모델 학습 ---
def train_improved_model(train_df, target_col='매출수량'):
    """개선된 모델 학습 (타겟 인코딩 + 고급 피처)"""
    
    print("개선된 피처 생성 중...")
    train_df, encoders = create_improved_features(train_df, is_train=True)
    
    # 타겟 인코딩 적용
    target_cols = ['영업장명', '메뉴명', '영업장명_메뉴명']
    dummy_test = train_df.head(1).copy()  # 더미 테스트 데이터
    train_df, _, target_encoders = safe_target_encoding(train_df, dummy_test, target_cols, target_col)
    
    # 피처 선택
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
    print(f"사용할 피처 수: {len(available_features)}")
    
    # 결측치 처리
    for col in available_features:
        if col.endswith('_encoded'):
            train_df[col] = train_df[col].fillna(-1)
        else:
            train_df[col] = train_df[col].fillna(0)
    
    # 시계열 교차 검증
    print("\n교차 검증 시작...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    X = train_df[available_features]
    y = train_df[target_col]
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1} 학습 중...")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # 개선된 CatBoost 모델
        model = CatBoostRegressor(
            iterations=500,  # 증가
            learning_rate=0.08,  # 약간 감소
            depth=7,  # 증가
            l2_leaf_reg=2,  # 감소
            bagging_temperature=1,  # 추가
            random_strength=1,  # 추가
            loss_function='RMSE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50
        )
        
        # 카테고리 피처 지정
        cat_features = [col for col in available_features if col.endswith('_encoded')]
        
        # 학습
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            cat_features=cat_features,
            use_best_model=True
        )
        
        # 예측 및 평가
        y_pred = model.predict(X_val_fold)
        y_pred = np.maximum(y_pred, 0)  # 음수 제거
        
        rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
        cv_scores.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    
    print(f"\n=== 교차 검증 결과 ===")
    print(f"CV 점수들: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"평균 RMSE: {np.mean(cv_scores):.4f}")
    print(f"표준편차: {np.std(cv_scores):.4f}")
    print(f"안정성 지수: {np.std(cv_scores)/np.mean(cv_scores):.3f}")
    
    # 성능 개선도 계산
    baseline_scores = [11.0336, 9.3732, 20.1483, 38.5346, 84.7470]  # 이전 결과
    improvement = np.mean(baseline_scores) - np.mean(cv_scores)
    print(f"베이스라인 대비 개선: {improvement:.4f} (RMSE 감소)")
    
    # 전체 데이터로 최종 모델 학습
    print("\n최종 모델 학습 중...")
    final_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.08,
        depth=7,
        l2_leaf_reg=2,
        bagging_temperature=1,
        random_strength=1,
        loss_function='RMSE',
        random_seed=42,
        verbose=100
    )
    
    final_model.fit(X, y, cat_features=cat_features)
    
    return final_model, encoders, available_features, target_encoders, train_df

# --- 개선된 예측 함수 ---
def predict_improved(test_df, model, encoders, feature_columns, target_encoders, train_df_for_target):
    """개선된 예측"""
    
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
    
    # 예측
    predictions = model.predict(test_df[feature_columns])
    predictions = np.maximum(predictions, 0)
    
    return predictions

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
    print("=== 개선된 모델 (타겟 인코딩 + 고급 피처) ===")
    print("주요 개선사항:")
    print(" 타겟 인코딩 (안전한 교차검증 방식)")
    print(" 삼각함수 기반 주기성 피처")
    print(" 상호작용 피처 (영업장×계절, 메뉴×요일)")
    print(" 고급 시간 피처 (월초/중/말, 급여일 등)")
    print(" 그룹별 이상치 처리")
    print(" 개선된 하이퍼파라미터")
    print()

    # 1. 데이터 로드
    print("데이터 로드 중...")
    train = pd.read_csv('./data/train/train.csv')
    print(f"훈련 데이터: {train.shape}")

    # 2. 개선된 모델 학습
    print("\n개선된 모델 학습 중...")
    model, encoders, feature_columns, target_encoders, processed_train = train_improved_model(train)

    # 3. 테스트 데이터 예측
    print("\n테스트 데이터 예측 중...")
    all_preds = []
    
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    for i, path in enumerate(test_files):
        print(f"처리 중 ({i+1}/{len(test_files)}): {os.path.basename(path)}")
        
        test_df = pd.read_csv(path)
        preds = predict_improved(test_df, model, encoders, feature_columns, target_encoders, processed_train)
        
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
    submission.to_csv('improved_target_encoding_submission.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== 완료 ===")
    print("예측 완료! 결과가 'improved_target_encoding_submission.csv'에 저장되었습니다.")
    print("\n예상 성능 향상:")
    print(" RMSE: 32.77 → 25-30 (약 15-25% 개선)")
    print(" 안정성: 0.854 → 0.6-0.7 (변동성 감소)")
    print(" 순위: 하위 50-70% → 중위 40-60%")
    print("\n다음 단계 개선 가능 영역:")
    print("외부 데이터 (날씨, 이벤트)")  
    print("앙상블 모델")
    print("하이퍼파라미터 최적화")