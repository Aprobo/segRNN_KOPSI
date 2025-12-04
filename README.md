# KOSPI Price Prediction (GRU + Time Series Modeling)

본 프로젝트는 **KOSPI 일별 데이터(Price, Volume, Fluctuation)** 를 이용해  
GRU 기반의 시계열 모델로 **다음 날 Price 예측하는 모델**입니다.

구현 포인트는 다음과 같습니다:

- Price, Volume, Fluct **각 feature 별 독립적 MinMax Scaling**
- segment_length (30일), look_back (7일) 기반 **슬라이딩 세그먼트 구조**
- PyTorch 기반 GRU 모델 (입력: 7×3 feature, 출력: 다음 날 Price)
- Train / Val / Test 자동 분리 및 RMSE 평가
- 예측값을 날짜 인덱스로 복원하여 시각화

---
## 📁 Dataset Structure

입력 CSV 파일 예시 (`kospi_pvf.csv`):

| Date       | Price  | Volume   | Fluct |
|------------|---------|----------|--------|
| 2010-01-04 | 1689.34 | 6533876  | 0.0032 |
| 2010-01-05 | 1696.32 | 7129372  | 0.0041 |
| ...        | ...     | ...      | ...    |

- `Price` : KOSPI 종가
- `Volume`: 거래량
- `Fluct` : 변동률 `(P_t - P_(t-1)) / P_(t-1)`

## Result
![overview](https://github.com/Aprobo/segRNN_KOPSI.git/result/p/kospi_test_prediction.png)
