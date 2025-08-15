# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # 한글 폰트 설정
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['axes.unicode_minus'] = False

# def analyze_train_data():
#     """train.csv 데이터 종합 분석"""
    
#     print("=" * 60)
#     print("🔍 리조트 식음업장 매출 데이터 분석 리포트")
#     print("=" * 60)
    
#     # 데이터 로드
#     print("\n📊 데이터 로딩 중...")
#     df = pd.read_csv('./data/train/train.csv')
    
#     # 날짜 변환
#     df['영업일자'] = pd.to_datetime(df['영업일자'])
#     df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
#     df['메뉴명'] = df['영업장명_메뉴명'].str.split('_').str[1:]
#     df['메뉴명'] = df['메뉴명'].apply(lambda x: '_'.join(x) if x else '')
    
#     # 1. 기본 정보
#     print("\n" + "="*50)
#     print("📈 1. 데이터 기본 정보")
#     print("="*50)
#     print(f"• 총 레코드 수: {len(df):,}건")
#     print(f"• 분석 기간: {df['영업일자'].min().strftime('%Y-%m-%d')} ~ {df['영업일자'].max().strftime('%Y-%m-%d')}")
#     print(f"• 총 분석 일수: {(df['영업일자'].max() - df['영업일자'].min()).days + 1}일")
#     print(f"• 영업장 수: {df['영업장명'].nunique()}개")
#     print(f"• 메뉴 수: {df['영업장명_메뉴명'].nunique()}개")
#     print(f"• 총 매출수량: {df['매출수량'].sum():,}개")
#     print(f"• 평균 매출수량: {df['매출수량'].mean():.2f}개")
    
#     # 2. 매출수량 분포 분석
#     print("\n" + "="*50)
#     print("📊 2. 매출수량 분포 분석")
#     print("="*50)
#     print(f"• 최소값: {df['매출수량'].min()}개")
#     print(f"• 최대값: {df['매출수량'].max()}개")
#     print(f"• 중앙값: {df['매출수량'].median()}개")
#     print(f"• 표준편차: {df['매출수량'].std():.2f}개")
#     print(f"• 매출 0인 비율: {(df['매출수량'] == 0).mean()*100:.1f}%")
#     print(f"• 매출 > 0인 평균: {df[df['매출수량'] > 0]['매출수량'].mean():.2f}개")
    
#     # 3. 영업장별 분석
#     print("\n" + "="*50)
#     print("🏪 3. 영업장별 매출 현황")
#     print("="*50)
#     store_analysis = df.groupby('영업장명')['매출수량'].agg(['sum', 'mean', 'count']).round(2)
#     store_analysis['비율(%)'] = (store_analysis['sum'] / store_analysis['sum'].sum() * 100).round(1)
#     store_analysis = store_analysis.sort_values('sum', ascending=False)
    
#     print("TOP 영업장 (총 매출수량 기준):")
#     for i, (store, data) in enumerate(store_analysis.head(10).iterrows(), 1):
#         print(f"{i:2d}. {store:15s}: {data['sum']:8,.0f}개 ({data['비율(%)']:5.1f}%) | 평균: {data['mean']:6.2f}개")
    
#     # 4. 인기 메뉴 분석
#     print("\n" + "="*50)
#     print("🍽️ 4. 인기 메뉴 분석")
#     print("="*50)
#     menu_analysis = df.groupby('메뉴명')['매출수량'].agg(['sum', 'mean', 'count']).round(2)
#     menu_analysis = menu_analysis.sort_values('sum', ascending=False)
    
#     print("TOP 10 인기 메뉴 (총 매출수량 기준):")
#     for i, (menu, data) in enumerate(menu_analysis.head(10).iterrows(), 1):
#         print(f"{i:2d}. {menu:25s}: {data['sum']:8,.0f}개 | 평균: {data['mean']:6.2f}개")
    
#     # 5. 월별 매출 추이
#     print("\n" + "="*50)
#     print("📅 5. 월별 매출 추이")
#     print("="*50)
#     df['연월'] = df['영업일자'].dt.to_period('M')
#     monthly_sales = df.groupby('연월')['매출수량'].sum()
    
#     print("월별 매출수량:")
#     for period, sales in monthly_sales.items():
#         print(f"• {period}: {sales:8,}개")
    
#     # 6. 요일별 매출 분석
#     print("\n" + "="*50)
#     print("📆 6. 요일별 매출 분석")
#     print("="*50)
#     df['요일'] = df['영업일자'].dt.dayofweek
#     weekday_names = ['월', '화', '수', '목', '금', '토', '일']
#     df['요일명'] = df['요일'].map(dict(enumerate(weekday_names)))
    
#     weekday_sales = df.groupby('요일명')['매출수량'].agg(['sum', 'mean']).round(2)
#     weekday_sales = weekday_sales.reindex(weekday_names)
    
#     print("요일별 매출 현황:")
#     for day, data in weekday_sales.iterrows():
#         print(f"• {day}요일: 총 {data['sum']:8,.0f}개 | 평균 {data['mean']:6.2f}개")
    
#     # 7. 영업장별 주력 메뉴
#     print("\n" + "="*50)
#     print("🎯 7. 영업장별 주력 메뉴")
#     print("="*50)
#     for store in df['영업장명'].unique():
#         store_data = df[df['영업장명'] == store]
#         top_menu = store_data.groupby('메뉴명')['매출수량'].sum().sort_values(ascending=False).head(3)
#         print(f"\n• {store}:")
#         for i, (menu, sales) in enumerate(top_menu.items(), 1):
#             print(f"  {i}. {menu}: {sales:,}개")
    
#     # 8. 계절성 분석
#     print("\n" + "="*50)
#     print("🌸 8. 계절성 분석")
#     print("="*50)
#     df['월'] = df['영업일자'].dt.month
#     df['계절'] = df['월'].map({12: '겨울', 1: '겨울', 2: '겨울',
#                           3: '봄', 4: '봄', 5: '봄',
#                           6: '여름', 7: '여름', 8: '여름',
#                           9: '가을', 10: '가을', 11: '가을'})
    
#     seasonal_sales = df.groupby('계절')['매출수량'].agg(['sum', 'mean']).round(2)
#     seasonal_order = ['봄', '여름', '가을', '겨울']
#     seasonal_sales = seasonal_sales.reindex(seasonal_order)
    
#     print("계절별 매출 현황:")
#     for season, data in seasonal_sales.iterrows():
#         print(f"• {season}: 총 {data['sum']:8,.0f}개 | 평균 {data['mean']:6.2f}개")
    
#     # 9. 특이사항 분석
#     print("\n" + "="*50)
#     print("⚠️ 9. 특이사항 분석")
#     print("="*50)
    
#     # 3월 매출 급감 현상
#     march_2023 = df[(df['영업일자'].dt.year == 2023) & (df['영업일자'].dt.month == 3)]['매출수량'].sum()
#     march_2024 = df[(df['영업일자'].dt.year == 2024) & (df['영업일자'].dt.month == 3)]['매출수량'].sum()
#     feb_2023 = df[(df['영업일자'].dt.year == 2023) & (df['영업일자'].dt.month == 2)]['매출수량'].sum()
#     feb_2024 = df[(df['영업일자'].dt.year == 2024) & (df['영업일자'].dt.month == 2)]['매출수량'].sum()
    
#     print(f"• 3월 매출 급감 현상:")
#     print(f"  - 2023년 2월 → 3월: {feb_2023:,} → {march_2023:,} ({(march_2023/feb_2023-1)*100:+.1f}%)")
#     print(f"  - 2024년 2월 → 3월: {feb_2024:,} → {march_2024:,} ({(march_2024/feb_2024-1)*100:+.1f}%)")
    
#     # 1월 매출 최고 현상
#     jan_2023 = df[(df['영업일자'].dt.year == 2023) & (df['영업일자'].dt.month == 1)]['매출수량'].sum()
#     jan_2024 = df[(df['영업일자'].dt.year == 2024) & (df['영업일자'].dt.month == 1)]['매출수량'].sum()
    
#     print(f"• 1월 매출 최고 현상:")
#     print(f"  - 2023년 1월: {jan_2023:,}개")
#     print(f"  - 2024년 1월: {jan_2024:,}개 (전체 최고)")
    
#     # 10. 데이터 품질 분석
#     print("\n" + "="*50)
#     print("🔍 10. 데이터 품질 분석")
#     print("="*50)
#     print(f"• 결측값:")
#     print(f"  - 영업일자: {df['영업일자'].isnull().sum()}개")
#     print(f"  - 영업장명_메뉴명: {df['영업장명_메뉴명'].isnull().sum()}개")
#     print(f"  - 매출수량: {df['매출수량'].isnull().sum()}개")
    
#     print(f"• 음수 매출수량: {(df['매출수량'] < 0).sum()}개")
#     print(f"• 이상치 (Q3 + 1.5*IQR 초과): {detect_outliers(df['매출수량'])}개")
    
#     # 11. 메뉴 카테고리 분석
#     print("\n" + "="*50)
#     print("🏷️ 11. 메뉴 카테고리 분석")
#     print("="*50)
    
#     # 메뉴 카테고리 분류
#     df['카테고리'] = '기타'
#     df.loc[df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False), '카테고리'] = '분식류'
#     df.loc[df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼|음료', na=False), '카테고리'] = '음료류'
#     df.loc[df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인|Beer|생맥주', na=False), '카테고리'] = '주류'
#     df.loc[df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥|갈비|공깃밥', na=False), '카테고리'] = '한식류'
#     df.loc[df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또|스파게티', na=False), '카테고리'] = '양식류'
#     df.loc[df['메뉴명'].str.contains('단체|패키지|세트|브런치', na=False), '카테고리'] = '단체메뉴'
#     df.loc[df['메뉴명'].str.contains('대여료|이용료|Conference|Convention', na=False), '카테고리'] = '대여료'
    
#     category_sales = df.groupby('카테고리')['매출수량'].agg(['sum', 'mean', 'count']).round(2)
#     category_sales['비율(%)'] = (category_sales['sum'] / category_sales['sum'].sum() * 100).round(1)
#     category_sales = category_sales.sort_values('sum', ascending=False)
    
#     print("카테고리별 매출 현황:")
#     for category, data in category_sales.iterrows():
#         print(f"• {category:8s}: {data['sum']:8,.0f}개 ({data['비율(%)']:5.1f}%) | 평균: {data['mean']:6.2f}개")
    
#     return df

# def detect_outliers(series):
#     """이상치 개수 계산 (IQR 방법)"""
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return ((series < lower_bound) | (series > upper_bound)).sum()

# def create_visualizations(df):
#     """시각화 생성"""
#     print("\n" + "="*50)
#     print("📊 12. 시각화 생성 중...")
#     print("="*50)
    
#     # 그래프 스타일 설정
#     plt.style.use('default')
#     fig, axes = plt.subplots(2, 3, figsize=(20, 12))
#     fig.suptitle('Resort Restaurant Sales Analysis', fontsize=16, fontweight='bold')
    
#     # 1. 월별 매출 추이
#     monthly_sales = df.groupby(df['영업일자'].dt.to_period('M'))['매출수량'].sum()
#     axes[0, 0].plot(range(len(monthly_sales)), monthly_sales.values, marker='o', linewidth=2, markersize=6)
#     axes[0, 0].set_title('Monthly Sales Trend', fontweight='bold')
#     axes[0, 0].set_xlabel('Month')
#     axes[0, 0].set_ylabel('Sales Quantity')
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].tick_params(axis='x', rotation=45)
    
#     # 2. 영업장별 매출
#     store_sales = df.groupby('영업장명')['매출수량'].sum().sort_values(ascending=True)
#     axes[0, 1].barh(range(len(store_sales)), store_sales.values, color='skyblue')
#     axes[0, 1].set_title('Sales by Store', fontweight='bold')
#     axes[0, 1].set_xlabel('Sales Quantity')
#     axes[0, 1].set_yticks(range(len(store_sales)))
#     axes[0, 1].set_yticklabels(store_sales.index, fontsize=8)
    
#     # 3. 요일별 매출
#     weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#     weekday_sales = df.groupby(df['영업일자'].dt.dayofweek)['매출수량'].sum()
#     axes[0, 2].bar(weekday_names, weekday_sales.values, color='lightgreen')
#     axes[0, 2].set_title('Sales by Day of Week', fontweight='bold')
#     axes[0, 2].set_xlabel('Day of Week')
#     axes[0, 2].set_ylabel('Sales Quantity')
    
#     # 4. 매출수량 분포
#     axes[1, 0].hist(df[df['매출수량'] > 0]['매출수량'], bins=50, color='orange', alpha=0.7, edgecolor='black')
#     axes[1, 0].set_title('Sales Quantity Distribution (>0)', fontweight='bold')
#     axes[1, 0].set_xlabel('Sales Quantity')
#     axes[1, 0].set_ylabel('Frequency')
#     axes[1, 0].set_yscale('log')
    
#     # 5. 카테고리별 매출
#     df['카테고리'] = '기타'
#     df.loc[df['메뉴명'].str.contains('꼬치어묵|떡볶이|핫도그|튀김|순대|어묵|파전', na=False), '카테고리'] = '분식류'
#     df.loc[df['메뉴명'].str.contains('아메리카노|라떼|콜라|스프라이트|생수|에이드|하이볼', na=False), '카테고리'] = '음료류'
#     df.loc[df['메뉴명'].str.contains('막걸리|소주|맥주|칵테일|와인', na=False), '카테고리'] = '주류'
#     df.loc[df['메뉴명'].str.contains('국밥|해장국|불고기|김치|된장|비빔밥', na=False), '카테고리'] = '한식류'
#     df.loc[df['메뉴명'].str.contains('파스타|피자|스테이크|샐러드|리조또', na=False), '카테고리'] = '양식류'
#     df.loc[df['메뉴명'].str.contains('단체|패키지|세트|브런치', na=False), '카테고리'] = '단체메뉴'
#     df.loc[df['메뉴명'].str.contains('대여료|이용료', na=False), '카테고리'] = '대여료'
    
#     category_sales = df.groupby('카테고리')['매출수량'].sum().sort_values(ascending=True)
#     colors = plt.cm.Set3(np.linspace(0, 1, len(category_sales)))
#     axes[1, 1].barh(range(len(category_sales)), category_sales.values, color=colors)
#     axes[1, 1].set_title('Sales by Category', fontweight='bold')
#     axes[1, 1].set_xlabel('Sales Quantity')
#     axes[1, 1].set_yticks(range(len(category_sales)))
#     axes[1, 1].set_yticklabels(category_sales.index, fontsize=9)
    
#     # 6. 상위 메뉴 매출
#     top_menus = df.groupby('메뉴명')['매출수량'].sum().sort_values(ascending=False).head(10)
#     axes[1, 2].barh(range(len(top_menus)), top_menus.values, color='coral')
#     axes[1, 2].set_title('Top 10 Menu Sales', fontweight='bold')
#     axes[1, 2].set_xlabel('Sales Quantity')
#     axes[1, 2].set_yticks(range(len(top_menus)))
#     axes[1, 2].set_yticklabels([menu[:20] + '...' if len(menu) > 20 else menu for menu in top_menus.index], fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig('train_data_analysis.png', dpi=300, bbox_inches='tight')
#     print("✅ 시각화 저장 완료: train_data_analysis.png")
    
#     return fig

# if __name__ == "__main__":
#     # 데이터 분석 실행
#     df = analyze_train_data()
    
#     # 시각화 생성
#     fig = create_visualizations(df)
    
#     print("\n" + "="*60)
#     print("✅ 분석 완료!")
#     print("📈 상세한 시각화는 'train_data_analysis.png' 파일을 확인하세요.")
#     print("="*60)


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_features(df):
    """
    매출 데이터에서 의미있는 새로운 특성들을 생성하는 함수
    """
    # 원본 데이터 복사
    enhanced_df = df.copy()
    
    # 1. 기본 전처리: 영업장명과 메뉴명 분리
    enhanced_df[['영업장명', '메뉴명']] = enhanced_df['영업장명_메뉴명'].str.split('_', expand=True)
    enhanced_df['영업일자'] = pd.to_datetime(enhanced_df['영업일자'])
    
    # 2. 시간 관련 특성 생성
    enhanced_df['연도'] = enhanced_df['영업일자'].dt.year
    enhanced_df['월'] = enhanced_df['영업일자'].dt.month
    enhanced_df['일'] = enhanced_df['영업일자'].dt.day
    enhanced_df['요일'] = enhanced_df['영업일자'].dt.dayofweek
    enhanced_df['요일명'] = enhanced_df['영업일자'].dt.day_name()
    
    # 계절 정의
    def get_season(month):
        if month in [3, 4, 5]:
            return '봄'
        elif month in [6, 7, 8]:
            return '여름'
        elif month in [9, 10, 11]:
            return '가을'
        else:
            return '겨울'
    
    enhanced_df['계절'] = enhanced_df['월'].apply(get_season)
    enhanced_df['주말여부'] = (enhanced_df['요일'].isin([5, 6])).astype(int)
    enhanced_df['월말여부'] = (enhanced_df['일'] >= 25).astype(int)
    enhanced_df['월초여부'] = (enhanced_df['일'] <= 7).astype(int)
    enhanced_df['분기'] = enhanced_df['영업일자'].dt.quarter
    
    # 3. 메뉴 카테고리 자동 분류
    def categorize_menu(menu_name):
        menu_lower = menu_name.lower()
        if any(keyword in menu_lower for keyword in ['수저', '젓가락', '포크', '나이프']):
            return '식기류'
        elif any(keyword in menu_lower for keyword in ['밥', '쌀', '면', '국수']):
            return '주식'
        elif any(keyword in menu_lower for keyword in ['고기', '삼겹', '갈비', '치킨', '돼지', '소고기']):
            return '육류'
        elif any(keyword in menu_lower for keyword in ['야채', '샐러드', '채소', '김치']):
            return '채소류'
        elif any(keyword in menu_lower for keyword in ['음료', '커피', '차', '주스', '물']):
            return '음료'
        elif any(keyword in menu_lower for keyword in ['디저트', '케이크', '아이스크림', '과자']):
            return '디저트'
        elif any(keyword in menu_lower for keyword in ['국', '찌개', '탕', '스프']):
            return '국물류'
        else:
            return '기타'
    
    enhanced_df['메뉴카테고리'] = enhanced_df['메뉴명'].apply(categorize_menu)
    
    # 4. 상호작용 특성 생성
    enhanced_df['영업장_카테고리'] = enhanced_df['영업장명'] + '_' + enhanced_df['메뉴카테고리']
    enhanced_df['계절_카테고리'] = enhanced_df['계절'] + '_' + enhanced_df['메뉴카테고리']
    enhanced_df['주말_카테고리'] = enhanced_df['주말여부'].map({0: '평일', 1: '주말'}) + '_' + enhanced_df['메뉴카테고리']
    
    # 5. 집계 통계 특성 생성
    # 영업장별 통계
    store_stats = enhanced_df.groupby('영업장명')['매출수량'].agg(['sum', 'mean', 'std', 'count']).reset_index()
    store_stats.columns = ['영업장명', '영업장_총매출', '영업장_평균매출', '영업장_매출표준편차', '영업장_데이터수']
    enhanced_df = enhanced_df.merge(store_stats, on='영업장명', how='left')
    
    # 메뉴별 통계
    menu_stats = enhanced_df.groupby('메뉴명')['매출수량'].agg(['sum', 'mean', 'std', 'count']).reset_index()
    menu_stats.columns = ['메뉴명', '메뉴_총매출', '메뉴_평균매출', '메뉴_매출표준편차', '메뉴_데이터수']
    enhanced_df = enhanced_df.merge(menu_stats, on='메뉴명', how='left')
    
    # 카테고리별 통계
    category_stats = enhanced_df.groupby('메뉴카테고리')['매출수량'].agg(['sum', 'mean', 'std']).reset_index()
    category_stats.columns = ['메뉴카테고리', '카테고리_총매출', '카테고리_평균매출', '카테고리_매출표준편차']
    enhanced_df = enhanced_df.merge(category_stats, on='메뉴카테고리', how='left')
    
    # 6. 매출 비중 계산
    enhanced_df['매출비중_영업장내'] = (enhanced_df['매출수량'] / enhanced_df['영업장_총매출'] * 100).fillna(0)
    enhanced_df['매출비중_메뉴내'] = (enhanced_df['매출수량'] / enhanced_df['메뉴_총매출'] * 100).fillna(0)
    enhanced_df['매출비중_카테고리내'] = (enhanced_df['매출수량'] / enhanced_df['카테고리_총매출'] * 100).fillna(0)
    
    # 7. 시계열 지연(Lag) 특성 생성
    enhanced_df = enhanced_df.sort_values(['영업장명', '메뉴명', '영업일자'])
    
    # 영업장-메뉴 조합별로 lag 특성 생성
    lag_features = []
    for (store, menu), group in enhanced_df.groupby(['영업장명', '메뉴명']):
        group = group.sort_values('영업일자').copy()
        
        # lag 특성들
        group['전일매출'] = group['매출수량'].shift(1).fillna(0)
        group['다음일매출'] = group['매출수량'].shift(-1).fillna(0)
        group['전주동요일매출'] = group['매출수량'].shift(7).fillna(0)
        
        # 이동 평균
        group['최근3일평균'] = group['매출수량'].rolling(window=3, min_periods=1).mean()
        group['최근7일평균'] = group['매출수량'].rolling(window=7, min_periods=1).mean()
        group['최근30일평균'] = group['매출수량'].rolling(window=30, min_periods=1).mean()
        
        # 변화율
        group['전일대비증감률'] = ((group['매출수량'] - group['전일매출']) / (group['전일매출'] + 1e-8) * 100).fillna(0)
        group['전주동요일대비증감률'] = ((group['매출수량'] - group['전주동요일매출']) / (group['전주동요일매출'] + 1e-8) * 100).fillna(0)
        
        lag_features.append(group)
    
    enhanced_df = pd.concat(lag_features, ignore_index=True)
    
    # 8. 추가 파생 특성
    # 매출 등급 (0: 무매출, 1: 저매출, 2: 보통, 3: 고매출)
    def sales_grade(sales):
        if sales == 0:
            return 0
        elif sales <= 5:
            return 1
        elif sales <= 20:
            return 2
        else:
            return 3
    
    enhanced_df['매출등급'] = enhanced_df['매출수량'].apply(sales_grade)
    
    # 트렌드 방향 (최근 7일 기준)
    enhanced_df['트렌드방향'] = np.where(
        enhanced_df['최근7일평균'] > enhanced_df['최근7일평균'].shift(7),
        '상승',
        np.where(enhanced_df['최근7일평균'] < enhanced_df['최근7일평균'].shift(7), '하락', '유지')
    )
    
    # 계절성 지수 (해당 월의 전체 평균 대비)
    monthly_avg = enhanced_df.groupby(['영업장명', '메뉴명', '월'])['매출수량'].mean().reset_index()
    monthly_avg.columns = ['영업장명', '메뉴명', '월', '월별평균매출']
    enhanced_df = enhanced_df.merge(monthly_avg, on=['영업장명', '메뉴명', '월'], how='left')
    
    overall_avg = enhanced_df.groupby(['영업장명', '메뉴명'])['매출수량'].mean().reset_index()
    overall_avg.columns = ['영업장명', '메뉴명', '전체평균매출']
    enhanced_df = enhanced_df.merge(overall_avg, on=['영업장명', '메뉴명'], how='left')
    
    enhanced_df['계절성지수'] = (enhanced_df['월별평균매출'] / (enhanced_df['전체평균매출'] + 1e-8)).fillna(1.0)
    
    # 9. 이상치 탐지 (Z-score 기준)
    enhanced_df['매출_zscore'] = enhanced_df.groupby(['영업장명', '메뉴명'])['매출수량'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    enhanced_df['이상치여부'] = (np.abs(enhanced_df['매출_zscore']) > 2).astype(int)
    
    # 10. 요일별 매출 패턴
    dow_pattern = enhanced_df.groupby(['영업장명', '메뉴명', '요일'])['매출수량'].mean().reset_index()
    dow_pattern.columns = ['영업장명', '메뉴명', '요일', '요일별평균매출']
    enhanced_df = enhanced_df.merge(dow_pattern, on=['영업장명', '메뉴명', '요일'], how='left')
    
    # 11. 연속 무매출 일수
    def count_consecutive_zeros(group):
        group = group.sort_values('영업일자')
        consecutive_zeros = []
        current_count = 0
        
        for sales in group['매출수량']:
            if sales == 0:
                current_count += 1
            else:
                current_count = 0
            consecutive_zeros.append(current_count)
        
        group['연속무매출일수'] = consecutive_zeros
        return group
    
    enhanced_df = enhanced_df.groupby(['영업장명', '메뉴명']).apply(count_consecutive_zeros).reset_index(drop=True)
    
    # 12. 유니크 식별자
    enhanced_df['유니크키'] = enhanced_df['영업장명'] + '_' + enhanced_df['메뉴명'] + '_' + enhanced_df['영업일자'].dt.strftime('%Y%m%d')
    
    # 정렬
    enhanced_df = enhanced_df.sort_values(['영업장명', '메뉴명', '영업일자']).reset_index(drop=True)
    
    return enhanced_df

# 사용 예시
def main():
    """
    메인 실행 함수
    """
    # 데이터 로드 (실제 파일 경로로 변경)
    df = pd.read_csv('./data/train/train.csv')
    
    print("=== 원본 데이터 정보 ===")
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    print("\n=== 특성 생성 시작 ===")
    
    # 새로운 특성들 생성
    enhanced_df = create_enhanced_features(df)
    
    print(f"\n=== 결과 ===")
    print(f"확장된 데이터 크기: {enhanced_df.shape}")
    print(f"추가된 컬럼 수: {enhanced_df.shape[1] - df.shape[1]}")
    
    print(f"\n=== 새로 생성된 주요 특성들 ===")
    new_features = [
        '요일명', '계절', '주말여부', '메뉴카테고리', '영업장_카테고리',
        '영업장_평균매출', '메뉴_평균매출', '매출비중_영업장내',
        '전일매출', '최근7일평균', '전일대비증감률', '매출등급',
        '트렌드방향', '계절성지수', '이상치여부', '연속무매출일수'
    ]
    
    for feature in new_features:
        if feature in enhanced_df.columns:
            print(f"✅ {feature}")
    
    # 결과 저장
    enhanced_df.to_csv('enhanced_train.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 확장된 데이터가 'enhanced_train.csv'로 저장되었습니다.")
    
    # 샘플 확인
    print(f"\n=== 샘플 데이터 ===")
    sample_cols = ['영업일자', '영업장명', '메뉴카테고리', '매출수량', '계절', '주말여부', '최근7일평균', '매출등급']
    print(enhanced_df[sample_cols].head(3).to_string())
    
    return enhanced_df

if __name__ == "__main__":
    enhanced_data = main()