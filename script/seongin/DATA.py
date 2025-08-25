import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_daily_store_sales():
    """
    각 매장별로 일일 메뉴의 총 매출량을 분석하는 함수
    """
    print("=" * 60)
    print("🏪 매장별 일일 총 매출량 분석")
    print("=" * 60)
    
    # 데이터 로드
    print("\n📊 데이터 로딩 중...")
    df = pd.read_csv('./data/train/train.csv')
    
    # 날짜 변환 및 영업장명 분리
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['영업장명'] = df['영업장명_메뉴명'].str.split('_').str[0]
    
    # 1. 매장별 일일 총 매출량 계산
    print("\n" + "="*50)
    print("📈 1. 매장별 일일 총 매출량")
    print("="*50)
    
    daily_store_sales = df.groupby(['영업일자', '영업장명'])['매출수량'].sum().reset_index()
    daily_store_sales = daily_store_sales.sort_values(['영업장명', '영업일자'])
    
    # 매장별 통계
    store_summary = daily_store_sales.groupby('영업장명').agg({
        '매출수량': ['sum', 'mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    store_summary.columns = ['총매출', '일평균매출', '일매출표준편차', '최소일매출', '최대일매출', '영업일수']
    store_summary = store_summary.sort_values('총매출', ascending=False)
    
    print("매장별 일일 매출 현황:")
    for store, data in store_summary.iterrows():
        print(f"• {store:15s}: 총 {data['총매출']:8,.0f}개 | 일평균 {data['일평균매출']:6.1f}개 | 최대 {data['최대일매출']:6.0f}개")
    
    # 2. 매장별 0매출 날짜 분석 (새로 추가)
    print("\n" + "="*50)
    print("🚫 2. 매장별 0매출 날짜 분석")
    print("="*50)
    
    # 0매출 날짜 필터링
    zero_sales_days = daily_store_sales[daily_store_sales['매출수량'] == 0].copy()
    zero_sales_days = zero_sales_days.sort_values(['영업장명', '영업일자'])
    
    # 매장별 0매출 통계
    zero_sales_summary = zero_sales_days.groupby('영업장명').agg({
        '영업일자': ['count', 'min', 'max']
    }).round(2)
    
    zero_sales_summary.columns = ['0매출일수', '첫0매출일', '마지막0매출일']
    zero_sales_summary = zero_sales_summary.sort_values('0매출일수', ascending=False)
    
    print("매장별 0매출 현황:")
    for store, data in zero_sales_summary.iterrows():
        total_days = store_summary.loc[store, '영업일수']
        zero_ratio = (data['0매출일수'] / total_days * 100)
        print(f"• {store:15s}: {data['0매출일수']:3.0f}일 ({zero_ratio:5.1f}%) | {data['첫0매출일'].strftime('%Y-%m-%d')} ~ {data['마지막0매출일'].strftime('%Y-%m-%d')}")
    
    # 3. 0매출 날짜 상세 분석
    print("\n" + "="*50)
    print("�� 3. 0매출 날짜 상세 분석")
    print("="*50)
    
    # 연속 0매출 구간 찾기
    def find_consecutive_zero_periods(store_data):
        """연속된 0매출 구간을 찾는 함수"""
        store_data = store_data.sort_values('영업일자')
        zero_periods = []
        start_date = None
        
        for idx, row in store_data.iterrows():
            if row['매출수량'] == 0:
                if start_date is None:
                    start_date = row['영업일자']
            else:
                if start_date is not None:
                    end_date = store_data.loc[store_data['영업일자'] < row['영업일자'], '영업일자'].iloc[-1]
                    duration = (end_date - start_date).days + 1
                    zero_periods.append({
                        '시작일': start_date,
                        '종료일': end_date,
                        '기간': duration
                    })
                    start_date = None
        
        # 마지막 구간 처리
        if start_date is not None:
            end_date = store_data['영업일자'].iloc[-1]
            duration = (end_date - start_date).days + 1
            zero_periods.append({
                '시작일': start_date,
                '종료일': end_date,
                '기간': duration
            })
        
        return zero_periods
    
    print("매장별 연속 0매출 구간:")
    for store in zero_sales_summary.index:
        store_zero_data = zero_sales_days[zero_sales_days['영업장명'] == store]
        if len(store_zero_data) > 0:
            periods = find_consecutive_zero_periods(store_zero_data)
            
            print(f"\n• {store}:")
            if periods:
                # 가장 긴 구간부터 정렬
                periods.sort(key=lambda x: x['기간'], reverse=True)
                for i, period in enumerate(periods[:5], 1):  # 상위 5개만 표시
                    print(f"  {i}. {period['시작일'].strftime('%Y-%m-%d')} ~ {period['종료일'].strftime('%Y-%m-%d')} ({period['기간']}일)")
            else:
                print("  연속 0매출 구간 없음")
    
    # 4. 요일별 0매출 분석
    print("\n" + "="*50)
    print("📆 4. 요일별 0매출 분석")
    print("="*50)
    
    zero_sales_days['요일'] = zero_sales_days['영업일자'].dt.dayofweek
    zero_sales_days['요일명'] = zero_sales_days['영업일자'].dt.day_name()
    
    weekday_zero_sales = zero_sales_days.groupby(['영업장명', '요일명']).size().reset_index(name='0매출횟수')
    weekday_zero_sales = weekday_zero_sales.pivot(index='영업장명', columns='요일명', values='0매출횟수').fillna(0)
    
    # 요일 순서 정렬
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_zero_sales = weekday_zero_sales.reindex(columns=weekday_order)
    
    print("요일별 0매출 횟수 (매장별):")
    print(weekday_zero_sales.astype(int).to_string())
    
    # 5. 월별 0매출 분석
    print("\n" + "="*50)
    print("📅 5. 월별 0매출 분석")
    print("="*50)
    
    zero_sales_days['연월'] = zero_sales_days['영업일자'].dt.to_period('M')
    monthly_zero_sales = zero_sales_days.groupby(['영업장명', '연월']).size().reset_index(name='0매출횟수')
    monthly_zero_sales = monthly_zero_sales.pivot(index='영업장명', columns='연월', values='0매출횟수').fillna(0)
    
    print("월별 0매출 횟수 (매장별):")
    print(monthly_zero_sales.astype(int).to_string())
    
    # 6. 전체 0매출 패턴 분석
    print("\n" + "="*50)
    print("📋 6. 전체 0매출 패턴 분석")
    print("="*50)
    
    # 모든 매장이 동시에 0매출인 날짜 찾기
    all_zero_days = daily_store_sales.groupby('영업일자')['매출수량'].sum()
    all_zero_dates = all_zero_days[all_zero_days == 0].index.tolist()
    
    print(f"• 모든 매장이 동시에 0매출인 날짜: {len(all_zero_dates)}일")
    if all_zero_dates:
        print("  주요 날짜들:")
        for date in sorted(all_zero_dates)[:10]:  # 상위 10개만 표시
            print(f"    - {date.strftime('%Y-%m-%d')}")
        if len(all_zero_dates) > 10:
            print(f"    ... 외 {len(all_zero_dates) - 10}일")
    
    # 7. 0매출 데이터 저장
    print("\n" + "="*50)
    print("📁 7. 0매출 데이터 저장")
    print("="*50)
    
    # 0매출 날짜 데이터 저장
    zero_sales_days.to_csv('zero_sales_days.csv', index=False, encoding='utf-8-sig')
    print("✅ 0매출 날짜 데이터가 'zero_sales_days.csv'로 저장되었습니다.")
    
    # 0매출 요약 데이터 저장
    zero_sales_summary.to_csv('zero_sales_summary.csv', encoding='utf-8-sig')
    print("✅ 0매출 요약 데이터가 'zero_sales_summary.csv'로 저장되었습니다.")
    
    # 8. 기존 분석 계속...
    print("\n" + "="*50)
    print("📅 8. 매장별 일일 매출 추이")
    print("="*50)
    
    # 각 매장별로 최고 매출일과 최저 매출일 찾기
    for store in store_summary.index:
        store_data = daily_store_sales[daily_store_sales['영업장명'] == store]
        
        max_day = store_data.loc[store_data['매출수량'].idxmax()]
        min_day = store_data.loc[store_data['매출수량'].idxmin()]
        
        print(f"\n• {store}:")
        print(f"  • 최고 매출일: {max_day['영업일자'].strftime('%Y-%m-%d')} ({max_day['매출수량']:,.0f}개)")
        print(f"  • 최저 매출일: {min_day['영업일자'].strftime('%Y-%m-%d')} ({min_day['매출수량']:,.0f}개)")
    
    # 9. 요일별 매장 매출 분석
    print("\n" + "="*50)
    print("📆 9. 요일별 매장 매출 분석")
    print("="*50)
    
    daily_store_sales['요일'] = daily_store_sales['영업일자'].dt.dayofweek
    daily_store_sales['요일명'] = daily_store_sales['영업일자'].dt.day_name()
    
    weekday_store_sales = daily_store_sales.groupby(['영업장명', '요일명'])['매출수량'].mean().reset_index()
    weekday_store_sales = weekday_store_sales.pivot(index='영업장명', columns='요일명', values='매출수량')
    
    # 요일 순서 정렬
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_store_sales = weekday_store_sales.reindex(columns=weekday_order)
    
    print("요일별 평균 매출 (매장별):")
    print(weekday_store_sales.round(1).to_string())
    
    # 10. 월별 매장 매출 분석
    print("\n" + "="*50)
    print("📅 10. 월별 매장 매출 분석")
    print("="*50)
    
    daily_store_sales['연월'] = daily_store_sales['영업일자'].dt.to_period('M')
    monthly_store_sales = daily_store_sales.groupby(['영업장명', '연월'])['매출수량'].sum().reset_index()
    monthly_store_sales = monthly_store_sales.pivot(index='영업장명', columns='연월', values='매출수량')
    
    print("월별 총 매출 (매장별):")
    print(monthly_store_sales.round(0).to_string())
    
    # 11. 매장별 매출 변동성 분석
    print("\n" + "="*50)
    print("📊 11. 매장별 매출 변동성 분석")
    print("="*50)
    
    # 변동계수 (CV = 표준편차/평균)
    store_summary['변동계수'] = (store_summary['일매출표준편차'] / store_summary['일평균매출']).round(3)
    store_summary['안정성등급'] = pd.cut(store_summary['변동계수'], 
                                    bins=[0, 0.5, 1.0, 2.0, float('inf')],
                                    labels=['매우안정', '안정', '불안정', '매우불안정'])
    
    print("매장별 매출 안정성:")
    for store, data in store_summary.iterrows():
        print(f"• {store:15s}: 변동계수 {data['변동계수']:5.3f} ({data['안정성등급']})")
    
    # 12. 매장별 성장률 분석
    print("\n" + "="*50)
    print("📈 12. 매장별 성장률 분석")
    print("="*50)
    
    # 분기별 매출 계산
    daily_store_sales['분기'] = daily_store_sales['영업일자'].dt.quarter
    daily_store_sales['연도'] = daily_store_sales['영업일자'].dt.year
    daily_store_sales['연도분기'] = daily_store_sales['연도'].astype(str) + 'Q' + daily_store_sales['분기'].astype(str)
    
    quarterly_sales = daily_store_sales.groupby(['영업장명', '연도분기'])['매출수량'].sum().reset_index()
    
    for store in store_summary.index:
        store_quarterly = quarterly_sales[quarterly_sales['영업장명'] == store].sort_values('연도분기')
        
        if len(store_quarterly) > 1:
            first_q = store_quarterly.iloc[0]['매출수량']
            last_q = store_quarterly.iloc[-1]['매출수량']
            growth_rate = ((last_q - first_q) / first_q * 100) if first_q > 0 else 0
            
            print(f"• {store:15s}: {first_q:8,.0f} → {last_q:8,.0f} ({growth_rate:+.1f}%)")
    
    # 13. 상세 일별 데이터 저장
    print("\n" + "="*50)
    print("📁 13. 상세 데이터 저장")
    print("="*50)
    
    # 일별 매장 매출 데이터 저장
    daily_store_sales.to_csv('daily_store_sales.csv', index=False, encoding='utf-8-sig')
    print("✅ 일별 매장 매출 데이터가 'daily_store_sales.csv'로 저장되었습니다.")
    
    # 매장별 요약 데이터 저장
    store_summary.to_csv('store_sales_summary.csv', encoding='utf-8-sig')
    print("✅ 매장별 요약 데이터가 'store_sales_summary.csv'로 저장되었습니다.")
    
    # 14. 주요 통계 요약
    print("\n" + "="*50)
    print("📋 14. 주요 통계 요약")
    print("="*50)
    
    print(f"• 분석 기간: {daily_store_sales['영업일자'].min().strftime('%Y-%m-%d')} ~ {daily_store_sales['영업일자'].max().strftime('%Y-%m-%d')}")
    print(f"• 총 분석 일수: {daily_store_sales['영업일자'].nunique()}일")
    print(f"• 매장 수: {daily_store_sales['영업장명'].nunique()}개")
    print(f"• 총 매출량: {daily_store_sales['매출수량'].sum():,}개")
    print(f"• 일평균 총 매출: {daily_store_sales.groupby('영업일자')['매출수량'].sum().mean():,.1f}개")
    
    # 최고/최저 매출일
    total_daily = daily_store_sales.groupby('영업일자')['매출수량'].sum()
    max_total_day = total_daily.idxmax()
    min_total_day = total_daily.idxmin()
    
    print(f"• 전체 최고 매출일: {max_total_day.strftime('%Y-%m-%d')} ({total_daily[max_total_day]:,.0f}개)")
    print(f"• 전체 최저 매출일: {min_total_day.strftime('%Y-%m-%d')} ({total_daily[min_total_day]:,.0f}개)")
    
    return daily_store_sales, store_summary, zero_sales_days, zero_sales_summary

def create_daily_sales_visualization(daily_store_sales, store_summary, zero_sales_days):
    """
    일별 매장 매출 시각화 생성
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n" + "="*50)
        print("📊 15. 시각화 생성")
        print("="*50)
        
        # 그래프 스타일 설정
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('매장별 일일 매출량 분석 (0매출 포함)', fontsize=16, fontweight='bold')
        
        # 1. 매장별 총 매출량
        store_summary['총매출'].plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('매장별 총 매출량', fontweight='bold')
        axes[0,0].set_xlabel('매장명')
        axes[0,0].set_ylabel('총 매출량')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 매장별 일평균 매출량
        store_summary['일평균매출'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('매장별 일평균 매출량', fontweight='bold')
        axes[0,1].set_xlabel('매장명')
        axes[0,1].set_ylabel('일평균 매출량')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 매장별 0매출 일수
        zero_sales_count = zero_sales_days.groupby('영업장명').size()
        zero_sales_count.plot(kind='bar', ax=axes[0,2], color='red', alpha=0.7)
        axes[0,2].set_title('매장별 0매출 일수', fontweight='bold')
        axes[0,2].set_xlabel('매장명')
        axes[0,2].set_ylabel('0매출 일수')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. 시간별 매출 추이 (전체)
        total_daily = daily_store_sales.groupby('영업일자')['매출수량'].sum()
        axes[1,0].plot(total_daily.index, total_daily.values, linewidth=2, color='red')
        axes[1,0].set_title('전체 일별 매출 추이', fontweight='bold')
        axes[1,0].set_xlabel('날짜')
        axes[1,0].set_ylabel('총 매출량')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. 매장별 변동계수
        store_summary['변동계수'].plot(kind='bar', ax=axes[1,1], color='orange')
        axes[1,1].set_title('매장별 매출 변동성 (변동계수)', fontweight='bold')
        axes[1,1].set_xlabel('매장명')
        axes[1,1].set_ylabel('변동계수')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. 0매출 날짜 분포
        zero_sales_days['월'] = zero_sales_days['영업일자'].dt.month
        monthly_zero_count = zero_sales_days.groupby('월').size()
        monthly_zero_count.plot(kind='bar', ax=axes[1,2], color='purple', alpha=0.7)
        axes[1,2].set_title('월별 0매출 발생 횟수', fontweight='bold')
        axes[1,2].set_xlabel('월')
        axes[1,2].set_ylabel('0매출 횟수')
        
        plt.tight_layout()
        plt.savefig('daily_store_sales_analysis_with_zero.png', dpi=300, bbox_inches='tight')
        print("✅ 시각화가 'daily_store_sales_analysis_with_zero.png'로 저장되었습니다.")
        
        return fig
        
    except ImportError:
        print("⚠️ matplotlib 또는 seaborn이 설치되지 않아 시각화를 건너뜁니다.")
        return None

def main():
    """
    메인 실행 함수
    """
    print("🏪 매장별 일일 매출량 분석 (0매출 포함)을 시작합니다...")
    
    # 일별 매장 매출 분석
    daily_store_sales, store_summary, zero_sales_days, zero_sales_summary = analyze_daily_store_sales()
    
    # 시각화 생성
    fig = create_daily_sales_visualization(daily_store_sales, store_summary, zero_sales_days)
    
    print("\n" + "="*60)
    print("✅ 매장별 일일 매출량 분석 (0매출 포함) 완료!")
    print("📊 상세한 시각화는 'daily_store_sales_analysis_with_zero.png' 파일을 확인하세요.")
    print("📁 데이터 파일:")
    print("   - daily_store_sales.csv: 일별 매장 매출 데이터")
    print("   - store_sales_summary.csv: 매장별 요약 통계")
    print("   - zero_sales_days.csv: 0매출 날짜 데이터")
    print("   - zero_sales_summary.csv: 0매출 요약 통계")
    print("="*60)
    
    return daily_store_sales, store_summary, zero_sales_days, zero_sales_summary

if __name__ == "__main__":
    daily_sales, summary, zero_days, zero_summary = main()