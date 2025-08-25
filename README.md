<img width="1192" height="225" alt="{BCD3D6E2-D455-4EAC-98B2-E8EE7D5A1398}" src="https://github.com/user-attachments/assets/a6520ca9-2ea4-40da-98ae-2d00ddf1fa6a" />



## Project Overview
The project aims to predict weekly demand for resort restaurant menu items using advanced machine learning techniques developed for the LG Aimers Hackathon.

## Final Performance
**Private Score: 0.51864 (Top 16.7%)**


## 👥 Team Members
<table>
  <tr align="center">
    <td width="150px">
      <a href="https://github.com/isshoman123" target="_blank">
        <img src="https://avatars.githubusercontent.com/isshoman123" alt="isshoman123" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/YeJin0217" target="_blank">
        <img src="https://avatars.githubusercontent.com/YeJin0217" alt="" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/dongsinwoo" target="_blank">
        <img src="https://avatars.githubusercontent.com/dongsinwoo" alt="dongsinwoo" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/espada105" target="_blank">
        <img src="https://avatars.githubusercontent.com/espada105" alt="espada105" />
      </a>
    </td>
  </tr>
  <tr align="center">
    <td>
      김재원
    </td>
    <td>
      박예진
    </td>
    <td>
      신동우
    </td>
      <td>
      홍성인
    </td>
  </tr>
</table>

## 🏗️ Technical Implementation

### 📊 Data Structure
- **Training Data**: Historical sales data for 9 resort venues with menu-specific demand patterns
- **Test Data**: 10 test files (TEST_00 ~ TEST_09) requiring 7-day forecasts
- **Venues**: 담하, 미라시아, 화담숲주막, 라그로타, 느티나무 셀프BBQ, 연회장, 카페테리아, 포레스트릿, 화담숲카페

### 🧠 Model Architecture

#### 1. **Direct Multi-Horizon Forecasting**
- 7개의 개별 모델을 각 예측 기간(1~7일)에 대해 훈련
- 각 모델은 해당 기간의 수요 패턴에 특화
- MLJAR AutoML을 사용하여 LightGBM과 CatBoost 알고리즘 자동 선택

#### 2. **Feature Engineering**

**시간 관련 특성:**
- 기본 시간 특성: year, month, day, dayofweek, quarter, weekofyear
- 순환 인코딩: month_sin/cos, dow_sin/cos, weekofyear_sin/cos
- One-hot 인코딩: 요일별, 월별 더미 변수
- 공휴일 특성: 공휴일 여부, 전후일 공휴일 효과

**시계열 특성:**
- 지연 특성: lag_1, lag_7, lag_14, lag_21, lag_28
- 이동평균: ma_7, ma_14, ma_28 (전체 및 양수값만)
- 지수이동평균: ewm_7, ewm_14
- 동일 요일 평균: dow_ma_4 (최근 4주)
- 추세 특성: trend_7, change_rate_7
- 희소성 특성: nonzero_rate_28, days_since_last_sale

**정적 특성:**
- 메뉴 라이프사이클: first_sale_month, peak_month, is_new_menu, is_discontinued
- 영업장별 가중치 적용

#### 3. **Ensemble Strategy**

**메타 블렌딩:**
```python
yhat = w_model * yhat_model + w_dow * dow_avg + w_ma7 * ma7
```
- 모델 예측값 (70%)
- 동일 요일 평균 (20%)
- 최근 7일 평균 (10%)

**기간별 가중치 조정:**
- 단기 예측(1-3일): 모델 가중치 ≥ 65%
- 중기 예측(4-5일): 모델 가중치 ≥ 60%
- 장기 예측(6-7일): 모델 가중치 ≥ 55%

### 🎯 Business Logic & Constraints

#### 1. **하한선 (Floor) 설정**
- 음료류: 최근 14일 내 판매 이력이 있거나 요일 평균 ≥ 1인 경우 최소 1개 보장
- 기타 품목: 최근 평균의 15%를 하한선으로 설정

#### 2. **상한선 (Cap) 설정**
- 기본: 과거 95백분위수 × 1.15
- 변동계수 > 1.5인 경우: 90백분위수 × 1.15
- 최대값: 과거 최대값 × 1.20

#### 3. **품목별 특화 규칙**
- **희소 품목**: 대여료, 단체식 등은 과대 예측 방지 (90% 수축)
- **카페테리아 고위험 메뉴**: 짜장면, 돈까스 등은 희소하거나 요일 평균이 0인 경우 93% 수축
- **브런치**: 평일에는 92% 보수적 예측
- **연회장 시설**: Conference/Convention Hall은 최대 1개, Grand Ballroom/OPUS는 최대 2개

### 📈 Training Strategy

#### 1. **샘플 가중치**
- 영업장별 가중치 적용 (미라시아: 36.3%, 담하: 18.2% 등)
- 매출수량이 0인 샘플은 가중치 0으로 설정
- 훈련 시 최소 가중치 0.05 보장

#### 2. **목표 변수 변환**
- log1p 변환으로 안정적인 훈련
- 예측 시 expm1으로 역변환

#### 3. **검증 전략**
- 시간순 분할 (shuffle=False)
- 훈련:검증 = 9:1 비율
- 최근 56일 데이터로 블렌딩 가중치 학습

### 🔧 Key Features

1. **메뉴 라이프사이클 분석**: 신메뉴, 단종 메뉴 식별
2. **계절성 패턴 캡처**: 순환 인코딩과 One-hot 인코딩 조합
3. **공휴일 효과 모델링**: 공휴일 전후 영향 고려
4. **동적 블렌딩**: 기간별 최적 가중치 학습
5. **비즈니스 규칙 통합**: 도메인 지식을 통한 예측 보정

### 📁 File Structure
```
script/dongwoo/0824_mljar_05263_copy.py  # 메인 모델링 코드
data/
├── train/train.csv                      # 훈련 데이터
├── test/TEST_*.csv                      # 테스트 데이터
└── sample_submission.csv                # 제출 형식
```

### 🚀 Usage
```bash
# 의존성 설치
pip install mljar-supervised pandas numpy

# 모델 훈련 및 예측 실행
python script/dongwoo/0824_mljar_05263_copy.py
```
