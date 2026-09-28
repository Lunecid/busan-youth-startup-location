# streamlit run app_admin_dong_v4.py
# -*- coding: utf-8 -*-
"""
부산 행정동 생활·소비 대시보드 v4 (확장성 고려, 격자 미적용)
- 선택 순서: 연도 → 행정동
- Choropleth: GeoJSON 키 선택, 경계선 표시, 0-1 정규화
- 카테고리별(방문/거주/직장/건수/금액) 시간대 라인 차트 분리
- 성별 도넛 차트
- 금액 단위: 만원
- 전처리 캐시(dash_preprocess.py) 사용
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="부산 행정동 대시보드 v4", layout="wide")

st.title("📊 부산 행정동 생활·소비 대시보드 v4")
st.caption("연도 → 행정동 → 지도 → 요약/카테고리 상세 · 금액 단위=만원 · 격자 확장 가능(미적용)")

# --------------------
# Sidebar: Paths / Cache
# --------------------
DATA_CSV = st.sidebar.text_input("CSV 경로", value="통합_행정동_데이터_1st.csv")
CACHE_DIR = st.sidebar.text_input("캐시 경로", value="./cache")
BUSAN_GEOJSON = st.sidebar.text_input("행정동 GeoJSON 경로", value="data/busan_dong.geojson")

# Build/Load cache
from dash_preprocess import build_cache
cache_dir = Path(CACHE_DIR)
meta_path = cache_dir / "meta.json"
if st.sidebar.button("🔄 캐시 생성/갱신"):
    info = build_cache(DATA_CSV, CACHE_DIR)
    st.sidebar.success(f"완료: {info}")
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st.sidebar.info(f"Rows={meta['n_rows']} | Cols={meta['n_cols']} | Years={meta['years']} | Dongs={meta['n_dong']}")

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

# --------------------
# Year → Dong selection
# --------------------
years = sorted(pd.concat([df["연도"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_years = st.sidebar.multiselect("① 연도 선택", years, default=years)
if not sel_years:
    st.info("좌측에서 먼저 **연도**를 선택하세요.")
    st.stop()

dongs = sorted(pd.concat([df["행정동명"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_dongs = st.sidebar.multiselect("② 행정동 선택 (선택 안 하면 전체)", dongs, default=[])

# --------------------
# GeoJSON handling
# --------------------
gj = None
feature_keys = []
featureidkey = None
if BUSAN_GEOJSON and Path(BUSAN_GEOJSON).exists():
    try:
        gj = json.loads(Path(BUSAN_GEOJSON).read_text(encoding="utf-8"))
        if len(gj.get("features", [])) > 0:
            props = list(gj["features"][0].get("properties", {}).keys())
            guesses = ["HADM_CD","EMD_CD","ADM_DR_CD","adm_cd","code","행정동코드"]
            in_common = [k for k in guesses if k in props]
            feature_keys = in_common + [k for k in props if k not in in_common]
        featureidkey = st.sidebar.selectbox("GeoJSON의 동코드 속성 선택", feature_keys, index=0 if feature_keys else None)
    except Exception as e:
        st.sidebar.error(f"GeoJSON 읽기 오류: {e}")
else:
    st.sidebar.warning("GeoJSON이 지정되지 않으면 지도는 표/차트로 대체됩니다.")

# --------------------
# Helpers
# --------------------
def df_for_map(metric: str, source: str, years_sel, dongs_sel):
    """metric in {'visitors','residents','workers','trx_cnt','trx_amt'}; source in {'시간대','성별'}"""
    if source == "시간대(24h 평균)":
        src = h
        q = src[(src["연도"].isin(years_sel)) & (src["지표"]==metric)].copy()
        if dongs_sel: q = q[q["행정동명"].isin(dongs_sel)]
        df_map = q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")
    else:
        src = g
        q = src[(src["연도"].isin(years_sel)) & (src["지표"]==metric)].copy()
        if dongs_sel: q = q[q["행정동명"].isin(dongs_sel)]
        df_map = q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")
    if metric == "trx_amt" and not df_map.empty:
        df_map["value"] = df_map["value"] / 10000.0  # 만원
    return df_map

def kpis_for_dong(dong: str, years_sel):
    # 월 평균 이용금액(만원)
    amt = g[(g["행정동명"]==dong) & (g["연도"].isin(years_sel)) & (g["지표"]=="trx_amt")]["값"].mean()
    amt_10k = (amt/10000.0) if pd.notnull(amt) else np.nan
    # 일 평균 방문인구(시간대 평균 합)
    h_vis = h[(h["행정동명"]==dong) & (h["연도"].isin(years_sel)) & (h["지표"]=="visitors")]
    daily_vis = (h_vis.groupby("시간")["값"].mean().sum()) if not h_vis.empty else np.nan
    return amt_10k, daily_vis

def donut_gender(dong: str, metric: str, years_sel):
    data = g[(g["행정동명"]==dong) & (g["연도"].isin(years_sel)) & (g["지표"]==metric)]
    if data.empty:
        return go.Figure()
    dd = data.groupby("성별")["값"].mean().reset_index()
    if metric == "trx_amt":
        dd["값"] = dd["값"] / 10000.0
    fig = go.Figure(go.Pie(labels=dd["성별"], values=dd["값"], hole=0.55))
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    return fig

def hourly_line(dong: str, metric: str, years_sel):
    data = h[(h["행정동명"]==dong) & (h["연도"].isin(years_sel)) & (h["지표"]==metric)]
    if data.empty:
        return go.Figure()
    s = data.groupby("시간")["값"].mean().reindex(range(24), fill_value=0).reset_index()
    if metric == "trx_amt":
        s["값"] = s["값"] / 10000.0
    fig = px.line(s, x="시간", y="값", markers=True)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(dtick=1))
    return fig

# --------------------
# Choropleth
# --------------------
st.markdown("### 🗺 행정동 Choropleth")
colA, colB, colC = st.columns([1.2,1,1])
with colA:
    metric_map = st.selectbox("지도 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0)
with colB:
    source_map = st.selectbox("지도 집계 소스", ["시간대(24h 평균)", "성별(월 평균)"], index=0)
with colC:
    norm_on = st.checkbox("0-1 정규화", value=True)

map_df = df_for_map(metric_map, "시간대(24h 평균)" if "시간대" in source_map else "성별(월 평균)", sel_years, sel_dongs)

if gj is not None and featureidkey:
    if map_df.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
    else:
        map_df["행정동코드"] = map_df["행정동코드"].astype(str)
        color_s = map_df["value"].astype(float)
        if norm_on:
            denom = (color_s.max()-color_s.min()) or 1.0
            map_df["value_norm"] = (color_s-color_s.min())/denom
            color_col = "value_norm"
        else:
            color_col = "value"
        fig = px.choropleth_mapbox(
            map_df,
            geojson=gj,
            featureidkey=f"properties.{featureidkey}",
            locations="행정동코드",
            color=color_col,
            hover_name="행정동명",
            mapbox_style="carto-positron",
            center={"lat":35.18, "lon":129.07},
            zoom=9,
            opacity=0.80,
            color_continuous_scale="Viridis"
        )
        fig.update_traces(marker_line_width=1.0, marker_line_color="white")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("GeoJSON을 지정하고 올바른 속성키를 선택하세요.")
    st.dataframe(map_df.sort_values("value", ascending=False).head(20))

# --------------------
# Summary + Category charts (separate)
# --------------------
st.markdown("## 📌 선택 행정동 요약 & 카테고리별 시각화")
candidates = map_df["행정동명"].unique().tolist() if not map_df.empty else dongs
sel_one = st.selectbox("행정동 선택", ["(선택)"] + candidates, index=0)
if sel_one != "(선택)":
    k1, k2 = st.columns(2)
    amt_10k, daily_vis = kpis_for_dong(sel_one, sel_years)
    with k1:
        st.metric("월 평균 이용금액 (만원)", f"{amt_10k:,.1f}" if pd.notnull(amt_10k) else "-")
    with k2:
        st.metric("일 평균 방문인구 (명)", f"{daily_vis:,.0f}" if pd.notnull(daily_vis) else "-")

    # Category sections
    st.markdown("### 🚶 유동인구(방문) — 시간대")
    st.plotly_chart(hourly_line(sel_one, "visitors", sel_years), use_container_width=True)

    st.markdown("### 🏠 거주인구 — 시간대")
    st.plotly_chart(hourly_line(sel_one, "residents", sel_years), use_container_width=True)

    st.markdown("### 🧳 직장인구 — 시간대")
    st.plotly_chart(hourly_line(sel_one, "workers", sel_years), use_container_width=True)

    st.markdown("### 🧾 이용건수 — 시간대")
    st.plotly_chart(hourly_line(sel_one, "trx_cnt", sel_years), use_container_width=True)

    st.markdown("### 💴 이용금액 — 시간대 (만원)")
    st.plotly_chart(hourly_line(sel_one, "trx_amt", sel_years), use_container_width=True)

    st.markdown("### 🚻 성별 비중 (도넛)")
    metric_for_donut = st.radio("지표 선택", ["visitors","residents","workers","trx_cnt","trx_amt"], horizontal=True, index=0, key="donut_metric")
    st.plotly_chart(donut_gender(sel_one, metric_for_donut, sel_years), use_container_width=True)
else:
    st.info("상단에서 행정동을 선택하면 요약/카테고리 그래프가 표시됩니다.")

# --------------------
# [TODO] Grid expansion hooks (미적용)
# --------------------
# - predict_grid.parquet 로드 → grid Choropleth 탭 추가
# - 위험도 색상, support 기반 투명도, 클릭 상세 등
