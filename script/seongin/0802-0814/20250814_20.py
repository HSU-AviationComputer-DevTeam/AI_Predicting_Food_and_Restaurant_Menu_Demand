import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from pandas.api.types import is_object_dtype, CategoricalDtype

# --- 1. 설정 (Configuration) ---
class Config:
	"""
	프로젝트 설정 값을 관리하는 클래스
	"""
	# 데이터 경로
	data_path = './data/'
	train_path = os.path.join(data_path, 'train/train.csv')
	test_path_pattern = os.path.join(data_path, 'test/*.csv')
	submission_path = os.path.join(data_path, 'sample_submission.csv')
	# 향상된 피처가 포함된 학습 데이터 경로 (사용자 생성)
	enhanced_train_path = r'C:\GitHubRepo\AI_Forecasting_Food_and_Restaurant_Menu_Demand\script\seongin\enhanced_train.csv'
	
	# 모델 및 결과 경로
	model_dir = './AutogluonModels/global_model'
	submission_dir = './submissions'
	output_filename = 'submission_global_model.csv'
	
	# 시드
	seed = 42
	
	# AutoGluon 설정
	time_limit = 3600  # 1시간 학습
	presets = 'best_quality'  # 고품질 프리셋 사용
	eval_metric = 'root_mean_squared_error'  # 이상치에 덜 민감한 RMSE 사용

# --- 2. 유틸리티 함수 ---
def set_seed(seed):
	"""
	시드 고정 함수
	"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_holiday_df():
	"""
	공휴일 정보를 담은 데이터프레임을 반환
	"""
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
	holiday_df = pd.DataFrame(holiday_dates, columns=['영업일자'])
	holiday_df['is_holiday'] = 1
	return holiday_df

# --- 3. 피처 엔지니어링 ---
def create_features(df, holiday_df):
	"""
	원시 컬럼(영업일자, 영업장명_메뉴명, 매출수량)을 기반으로 최소/기본 피처를 생성합니다.
	향상된 학습 데이터(enhanced_train.csv)와의 완전 일치는 보장하지 않으며,
	예측 시 부족한 피처는 0으로 보완하여 정렬합니다.
	"""
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    
	# 1. 날짜 기반 피처
	df['year'] = df['영업일자'].dt.year
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
	df['day_of_week'] = df['영업일자'].dt.weekday
    df['week_of_year'] = df['영업일자'].dt.isocalendar().week.astype(int)
	df['quarter'] = df['영업일자'].dt.quarter
	df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
	
	# 주기성 피처
	df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
	df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
	df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
	df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
	
	# 공휴일 피처
	df = pd.merge(df, holiday_df, on='영업일자', how='left')
	df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
	
	# 메뉴/영업장 분리 및 간단 카테고리
        df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:].apply(lambda x: '_'.join(x) if x else '')
        df['분식류'] = df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False).astype(int)
        df['음료류'] = df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료|식혜', na=False, case=False).astype(int)
        df['주류'] = df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|beer|생맥주', na=False, case=False).astype(int)
        df['식사류'] = df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥|파스타|피자|스테이크|샐러드|리조또', na=False).astype(int)
        df['단체/대여'] = df['메뉴명'].str.contains('단체|패키지|세트|대여료|conference|convention', na=False, case=False).astype(int)
    
	# 시계열 피처 (간단 Lag/Rolling)
	df = df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    if '매출수량' in df.columns:
		grouped = df.groupby('영업장명_메뉴명')['매출수량']
		for lag in [7, 14, 21, 28]:
            df[f'lag_{lag}'] = grouped.shift(lag)
		shifted = grouped.shift(1)
		for window in [7, 14, 28]:
			df[f'rolling_mean_{window}'] = shifted.rolling(window, min_periods=1).mean()
			df[f'rolling_std_{window}'] = shifted.rolling(window, min_periods=1).std()
			df[f'rolling_min_{window}'] = shifted.rolling(window, min_periods=1).min()
			df[f'rolling_max_{window}'] = shifted.rolling(window, min_periods=1).max()
	
	# 후처리: 불필요 컬럼 제거 및 결측치 채우기
        df = df.drop(columns=['영업장명', '메뉴명'])
    df = df.fillna(0)
    
	# 범주형 지정(모델이 자동 인식하나, 명시적으로 지정)
	for col in ['year', 'month', 'day', 'day_of_week', 'week_of_year', 'quarter']:
		if col in df.columns:
			df[col] = df[col].astype('category')
    
    return df

# 예측 시 학습 피처에 정렬/보완
def ensure_aligned_features(df_row: pd.DataFrame, feature_names: list, feature_dtypes: dict) -> pd.DataFrame:
	"""
	단일 행 데이터프레임(df_row)을 학습 시 사용된 피처(feature_names)에 맞춰
	컬럼을 정렬하고, 누락 컬럼은 dtype 에 맞춰 채워 추가합니다.
	 - 범주형/문자열: 'unknown'
	 - 수치형: 0
	"""
	for col in feature_names:
		if col not in df_row.columns:
			dt = feature_dtypes.get(col, None)
			if dt is not None and (isinstance(dt, CategoricalDtype) or is_object_dtype(dt)):
				df_row[col] = 'unknown'
			else:
				df_row[col] = 0
	return df_row.reindex(columns=feature_names)

# --- 4. 데이콘 결과 리포트 출력 함수 ---
def print_dacon_summary(config, predictor, output_path, dataset_info: str):
	"""
	데이콘 대회 형식의 최종 요약 리포트를 출력합니다.
	"""
	leaderboard = predictor.leaderboard(silent=True)
	# AutoGluon의 leaderboard는 score가 '높을수록 좋은' 방향으로 표기됩니다.
	try:
		best_row = leaderboard.sort_values('score_val', ascending=False).iloc[0]
		best_model = best_row['model']
	except Exception:
		best_model = str(leaderboard.iloc[0]['model'])
	
	print("\n\n" + "="*70)
	print("🏆 식음업장 메뉴 수요 예측 AI 온라인 해커톤 : 최종 결과 리포트 🏆")
	print("="*70)
	
	print("\n▌ 대회 개요")
	print("────────────────────────────────")
	print("  • 대회명: LG Aimers | 식음업장 메뉴 수요 예측 AI 온라인 해커톤")
	print("  • 참가자: 홍성인")
	print("  • 제출 방식: 단일 글로벌 모델을 활용한 자동화된 예측 파이프라인")
	print(f"  • 사용 데이터셋: {dataset_info}")
	
	print("\n▌ 실행 결과 요약")
	print("────────────────────────────────")
	print(f"  • 모델 저장 경로: {config.model_dir}")
	print(f"  • 최종 제출 파일: {output_path}")
	print(f"  • 사용된 최상위 모델: {best_model}")
	print(f"  • 평가 지표 (Validation): {config.eval_metric}")
	
	print("\n▌ AutoGluon 모델 학습 리더보드 (Validation Set 기준)")
	print("────────────────────────────────")
	print(leaderboard[['model', 'score_val', 'fit_time', 'predict_time']])
	
	print("\n▌ 대회 수료 기준 (참고)")
	print("────────────────────────────────")
	print("  • Public Score ≤ 0.711046")
	print("  • Private Score ≤ 0.693935")
	print("\n  [참고] 위 리더보드의 'score_val'은 Validation 데이터셋에 대한 점수이며,")
	print("         DACON의 Public/Private Score와는 데이터셋이 달라 직접 비교는 어렵습니다.")
	
	print("\n" + "="*70)
	print("🎉 모든 파이프라인 실행이 성공적으로 완료되었습니다. 🎉")
	print("="*70)

# --- 5. 메인 실행 로직 ---
def main():
	config = Config()
	set_seed(config.seed)
	
	if os.path.exists(config.model_dir):
		print(f"기존 모델 폴더({config.model_dir})를 삭제합니다.")
		shutil.rmtree(config.model_dir)
	os.makedirs(config.submission_dir, exist_ok=True)
	
	print("데이터 로딩 중...")
	# 향상된 학습 데이터 사용
	train_df = pd.read_csv(config.enhanced_train_path)
	submission_df = pd.read_csv(config.submission_path)
	holiday_df = get_holiday_df()
	# 안전장치
	if '매출수량' in train_df.columns:
		train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
	
	print("학습 데이터 준비(이미 피처 엔지니어링 완료)...")
	# enhanced_train.csv는 이미 피처 생성이 완료되었다고 가정
	train_featured = train_df.copy()
	# 학습에 불필요한 명시적 드롭 (날짜만 제거, 라벨은 유지)
	if '영업일자' in train_featured.columns:
		train_featured = train_featured.drop(columns=['영업일자'])
	
	# 학습 피처 목록 및 dtype 저장
	if '매출수량' not in train_featured.columns:
		raise ValueError("학습 데이터에 '매출수량' 컬럼이 필요합니다.")
	used_feature_columns = [c for c in train_featured.columns if c != '매출수량']
	feature_dtypes = train_featured[used_feature_columns].dtypes.to_dict()
	
	print("AutoGluon 글로벌 모델 학습 시작...")
            predictor = TabularPredictor(
                label='매출수량',
		path=config.model_dir,
		eval_metric=config.eval_metric,
		problem_type='regression'
            ).fit(
		train_featured,
		time_limit=config.time_limit,
		presets=config.presets,
		ag_args_fit={'num_gpus': 0},
		hyperparameters={'GBM': {}, 'CAT': {}, 'XGB': {}, 'RF': {}, 'XT': {}}
	)
	
	print("모델 학습 완료. 리더보드:")
	print(predictor.leaderboard(silent=True))
	
	print("순환 예측 시작...")
	test_paths = sorted(glob.glob(config.test_path_pattern))
	all_predictions = []
	
	for path in tqdm(test_paths, desc="Test 파일별 순환 예측"):
            test_file_df = pd.read_csv(path)
            basename = os.path.basename(path).replace('.csv', '')
		unique_menus_in_test = test_file_df['영업장명_메뉴명'].unique()
		
		for menu_name in unique_menus_in_test:
			# 과거 데이터 구성(필요 최소 컬럼만 사용)
            historical_data = pd.concat([
				train_df[train_df['영업장명_메뉴명'] == menu_name][['영업일자','영업장명_메뉴명','매출수량']],
				test_file_df[test_file_df['영업장명_메뉴명'] == menu_name][['영업일자','영업장명_메뉴명','매출수량']]
            ]).copy()
            
            for i in range(7):
				last_date = pd.to_datetime(historical_data['영업일자']).max()
                next_date = last_date + pd.Timedelta(days=1)
                new_row = pd.DataFrame([{'영업일자': next_date, '영업장명_메뉴명': menu_name, '매출수량': np.nan}])
                
                combined_for_feature = pd.concat([historical_data, new_row], ignore_index=True)
				featured_data_min = create_features(combined_for_feature, holiday_df)
				X_test_row = featured_data_min.tail(1).drop(columns=['영업일자', '매출수량'])
                
				# 학습 피처에 맞춰 정렬/보완 (dtype-aware)
				X_test_aligned = ensure_aligned_features(X_test_row, used_feature_columns, feature_dtypes)
                
				pred = predictor.predict(X_test_aligned, model='WeightedEnsemble_L2').iloc[0]
				pred = max(0, round(pred))
                
                all_predictions.append({
                    '영업일자': f"{basename}+{i+1}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred
                })

                new_prediction_row = new_row.copy()
                new_prediction_row['매출수량'] = pred
                historical_data = pd.concat([historical_data, new_prediction_row], ignore_index=True)

	print("제출 파일 생성 중...")
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
		submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량')
		final_submission = submission_df.set_index('영업일자')
        final_submission.update(submission_pivot)
        final_submission.reset_index(inplace=True)
		final_submission = final_submission.fillna(0)
    else:
        final_submission = submission_df.copy()

	output_path = os.path.join(config.submission_dir, config.output_filename)
	final_submission.to_csv(output_path, index=False, encoding='utf-8-sig')
	
	# 데이콘 형식의 최종 리포트 출력
	dataset_info = f"enhanced_train.csv (columns={len(train_df.columns)})"
	print_dacon_summary(config, predictor, output_path, dataset_info)

if __name__ == "__main__":
	main()
