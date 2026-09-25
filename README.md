<div align="center">

# 청년 창업가를 위한 부산 상권 최적 입지 제안
### 행정동 상권 유형화(K-Means)와 매출 예측(LightGBM)으로 찾은 "잠재력이 남은 동네"

**부산대학교 데이터사이언스대학원 × 부산광역시 DatoryLab 프로젝트**

![Program](https://img.shields.io/badge/DatoryLab-부산광역시_협력_과제-0b5cad?style=flat-square)
![Period](https://img.shields.io/badge/기간-2025.03–2025.11-555?style=flat-square)
![Poster](https://img.shields.io/badge/BUSAN_DATA_WEEK-포스터_발표-555?style=flat-square)
<br>
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-EB6E1F?style=flat-square)
![QGIS](https://img.shields.io/badge/QGIS-589632?style=flat-square&logo=qgis&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

> **English summary** — Only 15.9% of Korean founders under 30 survive five years, and restaurants (22.8% five-year survival) are where young founders cluster. For all ~200 administrative districts (*dong*) of Busan, we combined floating-population, card-spending and restaurant data, engineered market features (competition intensity, floating-to-resident ratio, youth/middle-aged share, time-of-day population), **segmented commercial districts with K-Means (K = 4)**, and **predicted monthly restaurant sales with LightGBM (R² = 0.897 with a log target)**. The gap between predicted and actual sales ("Potential Gap") highlights districts whose conditions promise more than they currently earn. We turned the results into cluster-specific start-up support proposals.

---

## 1. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 질문 | 부산에서 청년이 음식점을 연다면, **어느 행정동이 조건에 비해 기회가 남아 있는가?** |
| 분석 단위 | 부산광역시 전체 행정동 (약 200개) |
| 목표 변수 | 행정동별 요식·유흥 업종 월평균 매출액 |
| 방법 | 파생변수 설계 → K-Means 상권 유형화 → XGBoost / LightGBM 매출 회귀 → Potential Gap |
| 핵심 결과 | 상권 **4유형** 도출 · LightGBM **R² 0.897** · 잠재력 상·하위 10개 동 |

## 2. 문제 정의

- 한국 창업기업의 5년 생존율은 **33.8%**로 OECD 평균(45.4%)보다 낮고, 30세 미만 청년 창업가는 **15.9%**에 그친다.
- 청년 창업이 몰리는 **숙박·음식점업**의 5년 생존율은 **22.8%**로, 다섯 곳 중 네 곳이 5년 안에 문을 닫는다.
- 부산은 성장 정체와 청년 인구 유출을 함께 겪고 있어 청년 창업의 성공이 곧 도시 문제다.
- 같은 메뉴와 역량을 가진 음식점이라도 **어느 동에서 시작하느냐**에 따라 성과가 달라진다. 그런데 지금의 입지 선정은 창업자의 감이나 단편적인 정보에 기대고 있다.

## 3. 데이터

| 구분 | 내용 |
|---|---|
| 생활인구 | 행정동별 성별·시간대별·연령대별 생활인구 (월별 일평균) |
| 소비매출 | 행정동별 성별·시간대별·업종(대분류)별·연령대별 소비매출 (월별 일평균) |
| 음식점업 | 행정동별 음식점 수, 업종(한식·중식·일식·카페 등)별 통합 데이터 |

출처: [부산 Big-데이터웨이브](https://data.busan.go.kr/), [공공데이터포털](https://www.data.go.kr/). 원본 데이터는 이용 조건에 따라 저장소에 포함하지 않았습니다.

**파생변수**

| 변수 | 의미 |
|---|---|
| 경쟁 강도 | 유동인구 1명당 음식점 수 — 상권의 경쟁 치열도 |
| 유동·주거 비율 | 상업지형과 주거지형을 가르는 지표 |
| 청년 / 중장년 유동인구 비율 | "젊은 상권"과 "직장인 상권" 구분 |
| 시간대별 유동인구 | 11–14시(점심), 17–20시(저녁), 22–01시(야간) |

## 4. 분석 방법

```mermaid
flowchart LR
    A["데이터 수집·통합<br/>행정동 단위 집계"] --> B["파생변수 설계<br/>경쟁 강도 · 시간대 유동인구"]
    B --> C["표준화 후 K-Means<br/>Elbow + Silhouette → K=4"]
    B --> D["매출 회귀<br/>XGBoost vs LightGBM"]
    D --> E["Potential Gap<br/>예측 매출 − 실제 매출"]
    C --> F["군집별 창업 전략<br/>정책 제안"]
    E --> F
```

1. **통합·전처리**: 모든 데이터를 행정동 단위로 집계·피벗하고, 거리 기반 군집을 위해 표준화했다.
2. **상권 유형화**: 소비·생활인구·경쟁 특성이 비슷한 동을 묶기 위해 K-Means를 썼고, Elbow와 Silhouette로 K=4를 정했다.
3. **매출 예측**: XGBoost와 LightGBM을 비교하고, 치우친 매출 분포를 로그 변환했다.
4. **Potential Gap**: 조건으로 예측한 매출에서 실제 매출을 뺐다. 값이 클수록 **조건에 비해 아직 덜 번 동네**다.

## 5. 결과

### 5-1. 상권 4유형

<table>
<tr>
<td width="50%"><img src="figures/01_kmeans_pca_clusters.jpg" alt="행정동을 주성분 2개 평면에 찍고 4개 군집으로 색칠한 산점도"></td>
<td width="50%"><img src="figures/02_cluster_zscore_heatmap.jpg" alt="군집별 핵심 변수 Z-score 히트맵. 군집 1은 야간 유동인구, 군집 2는 중장년 비율과 경쟁 강도가 높다."></td>
</tr>
<tr>
<td align="center"><sub>K-Means 결과 (PCA 2차원 투영)</sub></td>
<td align="center"><sub>군집별 핵심 변수 Z-score</sub></td>
</tr>
</table>

| 군집 | 유형 | 특징 | 제안 업종 |
|---|---|---|---|
| 0 | 일반 주거지역 상권 | 모든 지표가 평균 이하, 경쟁 약함 | 배달 전문점, 가족 단위 생활밀착형 식당 |
| 1 | 번화가·야간 상권 | 22–01시 유동인구 매우 높음, 청년 비율 높음 | 주점·바, 야식, 패스트푸드 |
| 2 | 중장년층 중심 상권 | 중장년 유동인구 비율 최고, 경쟁 강도 높음 | 건강식·전통 음식점 (차별화 필수) |
| 3 | 주거·교통 허브 상권 | 상주인구 대비 유동인구 비율만 높음 | 출퇴근 시간대 브런치 카페, 픽업·간편식 |

### 5-2. 매출 예측

| 모델 | MSE | R² |
|---|---:|---:|
| XGBoost | 6.38e+17 | 0.621 |
| LightGBM | 2.79e+17 | 0.834 |
| **LightGBM (로그 변환)** | **1.74e+17** | **0.897** |

### 5-3. Potential Gap 상·하위 10개 동

| 순위 | 조건 대비 매출 여유 (상위) | Gap (억 원) | 군집 | 조건 대비 매출 초과 (하위) | Gap (억 원) | 군집 |
|---:|---|---:|---:|---|---:|---:|
| 1 | 사하구 하단2동 | 8.91 | 1 | 기장군 정관읍 | −16.54 | 0 |
| 2 | 부산진구 부전1동 | 8.56 | 1 | 강서구 대저2동 | −6.48 | 2 |
| 3 | 북구 구포1동 | 7.33 | 0 | 강서구 명지2동 | −3.46 | 0 |
| 4 | 서구 충무동 | 5.78 | 0 | 강서구 녹산동 | −2.64 | 1 |
| 5 | 연제구 연산6동 | 4.60 | 3 | 연제구 거제3동 | −2.59 | 0 |
| 6 | 부산진구 개금1동 | 4.40 | 0 | 금정구 남산동 | −1.99 | 0 |
| 7 | 사상구 감전동 | 4.34 | 2 | 북구 화명3동 | −1.85 | 1 |
| 8 | 부산진구 당감1동 | 3.59 | 0 | 금정구 부곡2동 | −1.63 | 0 |
| 9 | 서구 부민동 | 3.16 | 0 | 금정구 선두구동 | −1.40 | 2 |
| 10 | 동래구 수민동 | 3.01 | 0 | 동래구 사직3동 | −1.35 | 0 |

**해석.** 유동인구가 많은 곳이 답이 아니었다. 창업 아이템(주점, 브런치 카페 등)과 목표 고객(20대, 40대 직장인 등)에 맞는 **상권 군집을 고르는 것**이 초기 안정화의 관건이다.

## 6. 정책 제안

1. **군집별 맞춤형 창업 지원**: 대학가 상권엔 저가형 분식·카페, 오피스 상권엔 점심 특화 메뉴 컨설팅, 주거 상권엔 배달 전문점을 지원하는 식으로 군집 특성에 맞춘 차등 지원.
2. **데이터 기반 입지 컨설팅**: 창업 지원 기관(부산창조경제혁신센터 등)에서 이 모델로 예비 창업자에게 입지와 메뉴를 과학적으로 추천.
3. **저성과 군집 활성화**: 구조적으로 불리한 군집에는 임대료 지원, 공동 주방 설비, 지역 특산물 식당 유치 인센티브를 선제적으로 제공.
4. **대시보드**: 희망 업종과 자본금을 넣으면 동별 예상 매출과 추천 입지를 지도로 보여주는 웹 대시보드 프로토타입(Streamlit)을 만들었다.

## 7. 한계와 다음 단계

- **데이터 대표성**: 매출 데이터가 모집단을 완전히 대표하지 않을 수 있고, 코로나19 같은 외부 충격을 통제하지 못했다.
- **정적 모델**: 한 시점의 스냅샷이라 도시 개발이나 상권 변화를 반영하지 못한다. 월·분기 시계열(LSTM, GRU)로 확장할 계획이다.
- **다른 업종·창업자 특성**: 도소매·교육 서비스로 넓히고, 창업자 경력·자본금·교육 이수 여부를 넣어 "사람 × 장소" 다층 모델로 발전시킬 수 있다.
- **정성 연구 병행**: 성공한 창업자 인터뷰로 수치 뒤의 이야기를 확인할 필요가 있다.

## 8. 나의 역할

- 데이터 수집·전처리와 행정동 단위 통합
- 파생변수 설계, 군집 분석과 매출 예측 모델링
- Streamlit 웹 대시보드 개발
- 보고서 작성, BUSAN DATA WEEK 포스터 제작

## 9. 산출물

| 파일 | 설명 |
|---|---|
| [`docs/report.pdf`](docs/report.pdf) | 1학기 연구 보고서 (11쪽). 표지는 개인정보가 있어 제외했다. |
| [`figures/`](figures/) | 군집 분석 그림 |

<details>
<summary><b>참고문헌</b></summary>

- 통계청 (2025). 2024년도 청년(15~29세) 고용 특징.
- 중소벤처기업부 (2025). 2024년 연간 창업기업동향.
- 국회예산정책처 (2017). 청년창업 지원사업 성과분석 및 역할제고 방안.
- 신혜영. Associations of COVID-19 and Local Economy Analyzed with Structured and Unstructured Data: A Case Study of Seongdong-gu, Seoul.
- 김지영·이예림. 부산지역 창업활동이 지역경제 성장과 실업률 저감에 미치는 영향 분석.
- 배은솔·윤갑식. 부산광역시 지식서비스업 창업의 입지결정 요인분석.

</details>

---

<sub>팀 프로젝트 산출물입니다. 팀원의 개인정보 보호를 위해 이름은 적지 않았습니다. · 문의: [GitHub @Lunecid](https://github.com/Lunecid)</sub>
