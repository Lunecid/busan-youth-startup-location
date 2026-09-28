# streamlit run app_admin_dong_v6_fixed.py
# -*- coding: utf-8 -*-
"""
부산 행정동 생활·소비 대시보드 v6 (정규화 패치 포함)

요구 반영:
1) 연도 선택: '토글(단일 선택)' radio
2) 지도: 경계선 + 그라데이션 동작
   - 지표(연속형): 방문/거주/직장/이용건수/이용금액
   - 성별(발산형): 남-여 차이(diff) 기준, 남초=파랑 / 여초=빨강
3) 그래프: '선택한 것만' 보이도록 체크박스 토글 (기본 표시 안 함)
4) 성별 도넛: 기준 명시(선택 연도·행정동·지표 월평균 비중), 남=파랑/여=빨강
5) 설명: 히트 타임(피크) 자동 탐지 + 다른 행정동 대비 순위/백분위
6) 색상: 카테고리별 고정 컬러
7) 레이아웃: 상단 Map+요약, 하단 선택 그래프 카드

추가 패치:
- CSV의 '행정동코드'와 GeoJSON의 코드 속성 간 **정규화 매칭** 로직 포함
  (숫자/문자 타입 차이, 공백/하이픈 제거, zfill로 길이 맞춤)

주의:
- 숫자 단위/스케일 변경 없음(요청사항). /10000, 포맷 쉼표 등 미적용.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --------------------
# 페이지/테마
# --------------------
st.set_page_config(page_title="부산 행정동 대시보드 v6 (fixed)", layout="wide")
st.title("📊 부산 행정동 생활·소비 대시보드 v6 (fixed)")
st.caption("연도 → 행정동 → 지도/요약 → 선택 그래프 · 숫자 단위/스케일 변경 없음 · 코드 정규화 매칭")

# --------------------
# 사이드바: 경로/캐시
# --------------------
DATA_CSV = st.sidebar.text_input("CSV 경로", value="통합_행정동_데이터_1st.csv")
CACHE_DIR = st.sidebar.text_input("캐시 경로", value="./cache")
BUSAN_GEOJSON = st.sidebar.text_input("GeoJSON 경로", value="busan_205.geojson")

# 전처리 캐시 생성 버튼
from dash_preprocess import build_cache
cache_dir = Path(CACHE_DIR)
meta_path = cache_dir / "meta.json"
if st.sidebar.button("🔄 캐시 생성/갱신"):
    info = build_cache(DATA_CSV, CACHE_DIR)
    st.sidebar.success(f"완료: {info}")

# 캐시 로드
@st.cache_data(show_spinner=False)
def load_tables(cache_dir: str):
    cache = Path(cache_dir)
    g = pd.read_parquet(cache / "gender.parquet") if (cache / "gender.parquet").exists() else pd.DataFrame()
    h = pd.read_parquet(cache / "hourly.parquet") if (cache / "hourly.parquet").exists() else pd.DataFrame()
    a = pd.read_parquet(cache / "age.parquet") if (cache / "age.parquet").exists() else pd.DataFrame()
    return g, h, a

g, h, a = load_tables(CACHE_DIR)
if g.empty and h.empty and a.empty:
    st.warning("좌측에서 CSV 경로를 지정하고 '캐시 생성/갱신'을 눌러주세요.")
    st.stop()

# 메타
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.sidebar.info(f"Rows={meta['n_rows']} | Cols={meta['n_cols']} | Years={meta['years']} | Dongs={meta['n_dong']}")
    except Exception:
        pass

# --------------------
# UI: 연도(토글) → 행정동
# --------------------
years = sorted(pd.concat([df["연도"] for df in [g,h,a] if not df.empty]).unique().tolist())
if not years:
    st.error("데이터에 유효한 '연도'가 없습니다.")
    st.stop()

sel_year = st.sidebar.radio("① 연도 선택 (토글·단일)", options=years, index=0, key="year_radio")

dongs = sorted(pd.concat([df["행정동명"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_dong = st.sidebar.selectbox("② 행정동 선택", options=["(선택)"] + dongs, index=0)

# --------------------
# GeoJSON 로드 + 속성키 선택 + 코드 정규화 준비
# --------------------
gj = None
featureidkey = None
feature_keys = []
target_len = None

if BUSAN_GEOJSON and Path(BUSAN_GEOJSON).exists():
    try:
        gj = json.loads(Path(BUSAN_GEOJSON).read_text(encoding="utf-8"))
        if len(gj.get("features", [])) > 0:
            props = list(gj["features"][0].get("properties", {}).keys())
            guesses = ["HADM_CD","EMD_CD","ADM_DR_CD","adm_cd","ADM_CD","code","행정동코드"]
            in_common = [k for k in guesses if k in props]
            feature_keys = in_common + [k for k in props if k not in in_common]
        featureidkey = st.sidebar.selectbox("GeoJSON의 동코드 속성 선택", feature_keys, index=0 if feature_keys else None)

        # GeoJSON 코드 샘플 길이 파악(문자열화)
        if featureidkey and len(gj.get("features", [])) > 0:
            sample_code = str(gj["features"][0]["properties"][featureidkey])
            target_len = len(sample_code)
    except Exception as e:
        st.sidebar.error(f"GeoJSON 읽기 오류: {e}")
else:
    st.sidebar.warning("GeoJSON이 없으면 지도는 표로 대체됩니다.")

def normalize_code_series(s: pd.Series, target_len: int | None):
    """
    코드 문자열 정규화:
    - 공백/하이픈 제거
    - 숫자만 남김
    - target_len이 주어지면 zfill로 길이 맞춤
    """
    out = s.astype(str).str.replace(r"\s|-", "", regex=True) \
                       .str.replace(r"\D", "", regex=True)
    if target_len:
        out = out.str.zfill(target_len)
    return out

# --------------------
# 컬러 팔레트(일관성)
# --------------------
COLOR_VISITORS = "#2ca02c"  # 녹색
COLOR_RESIDENTS = "#9467bd" # 보라
COLOR_WORKERS  = "#8c564b"  # 갈색
COLOR_TRX_CNT  = "#ff7f0e"  # 주황
COLOR_TRX_AMT  = "#17becf"  # 청록
COLOR_MALE     = "#1f77b4"  # 파랑
COLOR_FEMALE   = "#d62728"  # 빨강

# --------------------
# 헬퍼
# --------------------
def _filter_g(year: int, metric: str, dong: str | None = None):
    q = g[(g["연도"]==year) & (g["지표"]==metric)]
    if dong and dong != "(선택)":
        q = q[q["행정동명"]==dong]
    return q

def _filter_h(year: int, metric: str, dong: str | None = None):
    q = h[(h["연도"]==year) & (h["지표"]==metric)]
    if dong and dong != "(선택)":
        q = q[q["행정동명"]==dong]
    return q

def df_for_map_continuous(metric: str, year: int) -> pd.DataFrame:
    """연속형 지도(지표)용: 시간대 평균 기반(24h)"""
    q = _filter_h(year, metric, None)
    if q.empty:
        return pd.DataFrame(columns=["행정동코드","행정동명","value"])
    return q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")

def df_for_map_gender(metric: str, year: int) -> pd.DataFrame:
    """성별 발산형 지도: 남-여 차이(>0=남초, <0=여초)"""
    q = _filter_g(year, metric, None)
    if q.empty:
        return pd.DataFrame(columns=["행정동코드","행정동명","diff","male","female"])
    pivot = q.pivot_table(index=["행정동코드","행정동명"], columns="성별", values="값", aggfunc="mean").reset_index()
    for c in ["male","female"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot["diff"] = pivot["male"].fillna(0) - pivot["female"].fillna(0)
    return pivot[["행정동코드","행정동명","diff","male","female"]]

def rank_and_percentile(df_map: pd.DataFrame, value_col: str, dong_name: str):
    """선택 동의 순위/백분위를 계산"""
    if df_map.empty or dong_name not in df_map["행정동명"].values:
        return None, None
    tmp = df_map[["행정동명", value_col]].dropna().sort_values(value_col, ascending=False).reset_index(drop=True)
    tmp["rank"] = np.arange(1, len(tmp)+1)
    row = tmp[tmp["행정동명"]==dong_name].iloc[0]
    rank = int(row["rank"])
    if len(tmp) > 1:
        pct = 100.0 * (1.0 - (rank-1)/(len(tmp)-1))
    else:
        pct = 100.0
    return rank, pct

def hourly_line(dong: str, metric: str, year: int, color_hex: str, height=240):
    data = _filter_h(year, metric, dong)
    if data.empty:
        return go.Figure(), None
    s = data.groupby("시간")["값"].mean().reindex(range(24), fill_value=0).reset_index()
    fig = px.line(s, x="시간", y="값", markers=True)
    fig.update_traces(line_color=color_hex)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(dtick=1), height=height)
    peak_idx = int(s.loc[s["값"].idxmax(), "시간"])
    fig.add_vline(x=peak_idx, line_dash="dot")
    return fig, peak_idx

def donut_gender_small(dong: str, metric: str, year: int, height=170):
    data = _filter_g(year, metric, dong)
    if data.empty:
        return go.Figure()
    dd = data.groupby("성별")["값"].mean().reset_index()
    colors = [COLOR_MALE if lab=="male" else COLOR_FEMALE for lab in dd["성별"].tolist()]
    fig = go.Figure(go.Pie(labels=dd["성별"], values=dd["값"], hole=0.6))
    fig.update_traces(textinfo="percent", hoverinfo="label+value", showlegend=False, marker=dict(colors=colors))
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=height)
    return fig

# --------------------
# 상단 레이아웃: Map + Summary
# --------------------
left, right = st.columns([1.3, 1])

with left:
    st.subheader("🗺 지도")
    map_mode = st.radio("지도 유형", ["지표(연속형)", "성별(남/여 비교)"], horizontal=True, key="map_mode")

    # 정규화 유틸: GeoJSON의 코드 길이에 맞춤
    def add_norm_code(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return df_in
        df = df_in.copy()
        if target_len:
            df["행정동코드_norm"] = normalize_code_series(df["행정동코드"], target_len)
        else:
            # 최소한 문자열화
            df["행정동코드_norm"] = df["행정동코드"].astype(str)
        return df

    if map_mode == "지표(연속형)":
        metric_map = st.selectbox("지표 선택", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="metric_map_cont")
        df_map = df_for_map_continuous(metric_map, sel_year)
        df_map = add_norm_code(df_map)
        color_col = "value"

        if gj is not None and featureidkey:
            if df_map.empty:
                st.warning("조건에 맞는 데이터가 없습니다.")
            else:
                fig = px.choropleth_mapbox(
                    df_map,
                    geojson=gj,
                    featureidkey=f"properties.{featureidkey}",
                    locations="행정동코드_norm",
                    color=color_col,
                    hover_name="행정동명",
                    mapbox_style="carto-positron",
                    center={"lat":35.18, "lon":129.07},
                    zoom=9,
                    opacity=0.85,
                    color_continuous_scale="Viridis"
                )
                fig.update_traces(marker_line_width=1.2, marker_line_color="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GeoJSON을 지정하고 올바른 속성키를 선택하세요.")
            st.dataframe(df_map.sort_values(color_col, ascending=False).head(20))

    else:  # 성별(발산형)
        metric_gender = st.selectbox("성별 비교 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="metric_map_gender")
        df_map = df_for_map_gender(metric_gender, sel_year)
        df_map = add_norm_code(df_map)

        if gj is not None and featureidkey:
            if df_map.empty:
                st.warning("조건에 맞는 데이터가 없습니다.")
            else:
                fig = px.choropleth_mapbox(
                    df_map,
                    geojson=gj,
                    featureidkey=f"properties.{featureidkey}",
                    locations="행정동코드_norm",
                    color="diff",
                    hover_name="행정동명",
                    mapbox_style="carto-positron",
                    center={"lat":35.18, "lon":129.07},
                    zoom=9,
                    opacity=0.85,
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0.0
                )
                fig.update_traces(marker_line_width=1.2, marker_line_color="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GeoJSON을 지정하고 올바른 속성키를 선택하세요.")
            st.dataframe(df_map.sort_values("diff", ascending=False).head(20))

with right:
    st.subheader("📌 요약")
    if sel_dong == "(선택)":
        st.info("우측 아래 '행정동 선택'에서 동을 선택하면 요약이 표시됩니다.")
    else:
        # 지도 기준 지표(연속형/성별)에 맞춰 순위/백분위 계산
        if "metric_map_cont" in st.session_state and st.session_state.get("map_mode") == "지표(연속형)":
            base_metric = st.session_state["metric_map_cont"]
            base_df_map = df_for_map_continuous(base_metric, sel_year)
            base_df_map = base_df_map.rename(columns={"value":"score"})
        else:
            base_metric = st.session_state.get("metric_map_gender", "visitors")
            base_df_map = df_for_map_gender(base_metric, sel_year).rename(columns={"diff":"score"})
        rnk, pct = rank_and_percentile(base_df_map.rename(columns={"score":"value"}), "value", sel_dong)

        st.markdown(f"- **선택 연도:** {sel_year}")
        st.markdown(f"- **선택 행정동:** {sel_dong}")
        if rnk is not None:
            st.markdown(f"- **지도 기준 지표 순위:** {rnk}위 (상위 약 {pct:.1f}%)")

        st.markdown("**🚻 성별 비중 (기준: 선택 연도·행정동·지표의 월평균 값 비중)**")
        donut_metric = st.selectbox("도넛 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="donut_metric")
        donut = donut_gender_small(sel_dong, donut_metric, sel_year, height=160)
        st.plotly_chart(donut, use_container_width=True)

# --------------------
# 하단: '선택한 그래프만' 보여주는 섹션
# --------------------
st.markdown("---")
st.header("📈 선택 그래프")

opt_visitors = st.checkbox("유동인구(방문) 시간대 라인", value=False)
opt_residents = st.checkbox("거주인구 시간대 라인", value=False)
opt_workers  = st.checkbox("직장인구 시간대 라인", value=False)
opt_cnt      = st.checkbox("이용건수 시간대 라인", value=False)
opt_amt      = st.checkbox("이용금액 시간대 라인", value=False)

sel_dong_bottom = st.selectbox("행정동 선택(그래프 섹션)", options=["(선택)"] + dongs, index=0, key="dong_for_charts")
if sel_dong_bottom == "(선택)":
    st.info("그래프를 보려면 행정동을 선택하세요.")
    st.stop()

c1, c2 = st.columns(2)

def _card(title_md: str, fig, explain: str | None = None):
    st.markdown(f"#### {title_md}")
    st.plotly_chart(fig, use_container_width=True)
    if explain:
        st.caption(explain)

if opt_visitors:
    with c1:
        fig, peak = hourly_line(sel_dong_bottom, "visitors", sel_year, COLOR_VISITORS, height=260)
        df_cmp = df_for_map_continuous("visitors", sel_year)
        rnk, pct = rank_and_percentile(df_cmp, "value", sel_dong_bottom)
        explain = f"히트 타임: **{peak}시** · 다른 행정동 대비 **{rnk}위(상위 {pct:.1f}%)**" if rnk else f"히트 타임: **{peak}시**"
        _card("🚶 유동인구(방문)", fig, explain)

if opt_residents:
    with c2:
        fig, peak = hourly_line(sel_dong_bottom, "residents", sel_year, COLOR_RESIDENTS, height=260)
        df_cmp = df_for_map_continuous("residents", sel_year)
        rnk, pct = rank_and_percentile(df_cmp, "value", sel_dong_bottom)
        explain = f"히트 타임: **{peak}시** · 다른 행정동 대비 **{rnk}위(상위 {pct:.1f}%)**" if rnk else f"히트 타임: **{peak}시**"
        _card("🏠 거주인구", fig, explain)

if opt_workers:
    with c1:
        fig, peak = hourly_line(sel_dong_bottom, "workers", sel_year, COLOR_WORKERS, height=260)
        df_cmp = df_for_map_continuous("workers", sel_year)
        rnk, pct = rank_and_percentile(df_cmp, "value", sel_dong_bottom)
        explain = f"히트 타임: **{peak}시** · 다른 행정동 대비 **{rnk}위(상위 {pct:.1f}%)**" if rnk else f"히트 타임: **{peak}시**"
        _card("🧳 직장인구", fig, explain)

if opt_cnt:
    with c2:
        fig, peak = hourly_line(sel_dong_bottom, "trx_cnt", sel_year, COLOR_TRX_CNT, height=260)
        df_cmp = df_for_map_continuous("trx_cnt", sel_year)
        rnk, pct = rank_and_percentile(df_cmp, "value", sel_dong_bottom)
        explain = f"히트 타임: **{peak}시** · 다른 행정동 대비 **{rnk}위(상위 {pct:.1f}%)**" if rnk else f"히트 타임: **{peak}시**"
        _card("🧾 이용건수", fig, explain)

if opt_amt:
    with c1:
        fig, peak = hourly_line(sel_dong_bottom, "trx_amt", sel_year, COLOR_TRX_AMT, height=260)
        df_cmp = df_for_map_continuous("trx_amt", sel_year)
        rnk, pct = rank_and_percentile(df_cmp, "value", sel_dong_bottom)
        explain = f"히트 타임: **{peak}시** · 다른 행정동 대비 **{rnk}위(상위 {pct:.1f}%)**" if rnk else f"히트 타임: **{peak}시**"
        _card("💴 이용금액", fig, explain)

# 기준 안내
st.markdown("---")
st.markdown("**성별 도넛 기준**: 선택 연도·행정동·지표의 **월평균 값**에서 **남/여 비중**을 계산합니다.")
st.markdown("**성별 지도(발산형) 기준**: **남-여 차이(diff)**를 기준으로 파랑(남초)↔빨강(여초) 색상을 적용합니다. 0은 균형.")