
# streamlit run app_busan_dong_dash.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="부산 행정동 생활·소비 대시보드", layout="wide")

st.title("📊 부산 행정동 생활·소비 대시보드 (1st)")
st.caption("연도별 / 행정동별 / 성별 / 시간대 / 연령대 데이터를 토글로 탐색")

# ---- Paths & cache ----
DATA_CSV = st.sidebar.text_input("CSV 경로", value="/mnt/data/통합_행정동_데이터_1st.csv")
CACHE_DIR = st.sidebar.text_input("캐시 경로", value="./cache")
BUSAN_GEOJSON = st.sidebar.text_input("행정동 GeoJSON 경로 (선택)", value="data/busan_dong.geojson")

# Build/Load cache
from dash_preprocess import build_cache
cache_dir = Path(CACHE_DIR)
meta_path = cache_dir / "meta.json"
if st.sidebar.button("🔄 캐시 생성/갱신"):
    info = build_cache(DATA_CSV, CACHE_DIR)
    st.sidebar.success(f"완료: {info}")
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st.sidebar.info(f"Rows={meta['n_rows']}, Cols={meta['n_cols']}, Years={meta['years']}, Dongs={meta['n_dong']}")

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

# ---- Sidebar filters ----
years = sorted(pd.concat([df["연도"] for df in [g,h,a] if not df.empty]).unique().tolist())
dongs = sorted(pd.concat([df["행정동명"] for df in [g,h,a] if not df.empty]).unique().tolist())

sel_years = st.sidebar.multiselect("연도 선택", years, default=years)
sel_dongs = st.sidebar.multiselect("행정동 선택 (선택 안 하면 전체)", dongs, default=[])

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["🗺 지도(Choropleth)", "⏱ 시간대 Heatmap", "🚻 성별 비교", "👥 연령대 분석"])

# ---- Choropleth ----
with tab1:
    st.subheader("행정동 Choropleth")
    metric = st.selectbox("지표 선택", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0)
    normalize = st.checkbox("행정동 내 정규화 (0-1)", value=True)
    agg_src = st.radio("집계 소스", ["hourly", "gender"], horizontal=True)
    if agg_src == "hourly":
        src = h
        # 시간대 평균
        q = src[(src["연도"].isin(sel_years)) & (src["지표"]==metric)].copy()
        if sel_dongs:
            q = q[q["행정동명"].isin(sel_dongs)]
        df_map = q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")
    else:  # gender monthly 평균
        src = g
        q = src[(src["연도"].isin(sel_years)) & (src["지표"]==metric)].copy()
        if sel_dongs:
            q = q[q["행정동명"].isin(sel_dongs)]
        df_map = q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")
    if normalize and not df_map.empty:
        v = df_map["value"].astype(float)
        denom = (v.max() - v.min()) or 1.0
        df_map["value_norm"] = (v - v.min()) / denom
    else:
        df_map["value_norm"] = df_map["value"]
    st.dataframe(df_map.head(10))

    if BUSAN_GEOJSON and Path(BUSAN_GEOJSON).exists():
        import json
        gj = json.loads(Path(BUSAN_GEOJSON).read_text(encoding="utf-8"))
        # NOTE: featureidkey는 보유한 GeoJSON의 행정동 코드 속성명에 맞춰 수정하세요.
        fig = px.choropleth_mapbox(
            df_map, geojson=gj, featureidkey="properties.HADM_CD",
            locations="행정동코드", color="value_norm",
            hover_name="행정동명", mapbox_style="carto-positron",
            center={"lat":35.18, "lon":129.07}, zoom=9, opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("GeoJSON 경로가 없어서 지도 시각화는 표로 대체됩니다. 좌측 입력에 GeoJSON 경로를 추가하세요.")

# ---- Hourly Heatmap ----
with tab2:
    st.subheader("시간대 Heatmap")
    metric = st.selectbox("지표 선택 (시간대)", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="hour_metric")
    scope = h[h["연도"].isin(sel_years) & (h["지표"]==metric)].copy()
    if sel_dongs: scope = scope[scope["행정동명"].isin(sel_dongs)]
    agg = scope.groupby(["시간"])["값"].mean().reindex(range(24)).reset_index()
    fig = px.imshow(agg[["값"]].T, aspect="auto",
                    labels=dict(x="시간(0~23)", color="값"),
                    x=list(range(24)))
    st.plotly_chart(fig, use_container_width=True)

# ---- Gender Compare ----
with tab3:
    st.subheader("성별 비교 (월 평균)")
    metric = st.selectbox("지표 선택 (성별)", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="gender_metric")
    gg = g[g["연도"].isin(sel_years) & (g["지표"]==metric)].copy()
    if sel_dongs: gg = gg[gg["행정동명"].isin(sel_dongs)]
    bar = gg.groupby(["성별"])["값"].mean().reset_index()
    fig = px.bar(bar, x="성별", y="값", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# ---- Age bands ----
with tab4:
    st.subheader("연령대 분석")
    typ = st.radio("유형", ["amt","cnt"], horizontal=True)
    aa = a[a["연도"].isin(sel_years) & (a["유형"]==typ)].copy()
    if sel_dongs: aa = aa[aa["행정동명"].isin(sel_dongs)]
    bar = aa.groupby(["연령대"])["값"].mean().reset_index()
    fig = px.bar(bar, x="연령대", y="값", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
