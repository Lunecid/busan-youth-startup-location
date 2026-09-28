
# streamlit run app_busan_dong_dash_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="부산 행정동 대시보드 v2", layout="wide")

st.title("📊 부산 행정동 생활·소비 대시보드 v2")
st.caption("연도 → 행정동 순서로 선택하고, 지도를 클릭하면 요약 패널이 갱신됩니다.")

# --------------------
# Sidebar: paths
# --------------------
DATA_CSV = st.sidebar.text_input("CSV 경로", value="통합_행정동_데이터_1st.csv")
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

# --------------------
# Filters (Year -> Dong order)
# --------------------
years = sorted(pd.concat([df["연도"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_years = st.sidebar.multiselect("① 연도 선택", years, default=years if years else [])

if not sel_years:
    st.info("좌측에서 먼저 **연도**를 선택하세요.")
    st.stop()

dongs = sorted(pd.concat([df["행정동명"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_dongs = st.sidebar.multiselect("② 행정동 선택 (선택 안 하면 전체)", dongs, default=[])

# --------------------
# Metric selection
# --------------------
metric = st.sidebar.selectbox("지도 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0)
agg_src = st.sidebar.radio("지도 집계소스", ["hourly(24h 평균)", "gender(월 평균)"], horizontal=False, index=0)
norm_on = st.sidebar.checkbox("지도 값 0-1 정규화", value=True)

# --------------------
# Helpers
# --------------------
def map_df_for_metric(metric: str, years_sel, dongs_sel):
    if "hourly" in agg_src:
        src = h
        q = src[(src["연도"].isin(years_sel)) & (src["지표"]==metric)].copy()
        if dongs_sel: q = q[q["행정동명"].isin(dongs_sel)]
        df_map = q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")
    else:
        src = g
        q = src[(src["연도"].isin(years_sel)) & (src["지표"]==metric)].copy()
        if dongs_sel: q = q[q["행정동명"].isin(dongs_sel)]
        df_map = q.groupby(["행정동코드","행정동명"])["값"].mean().reset_index(name="value")
    if df_map.empty:
        return df_map
    v = df_map["value"].astype(float)
    if norm_on:
        denom = (v.max() - v.min()) or 1.0
        df_map["value_norm"] = (v - v.min()) / denom
    else:
        df_map["value_norm"] = v
    return df_map

def compute_kpis(dong_name: str, years_sel):
    # 월 평균 이용금액
    amt = g[(g["행정동명"]==dong_name) & (g["연도"].isin(years_sel)) & (g["지표"]=="trx_amt")]["값"].mean()
    # 일 평균 방문인구(시간대 합계의 평균)
    h_vis = h[(h["행정동명"]==dong_name) & (h["연도"].isin(years_sel)) & (h["지표"]=="visitors")]
    daily_vis = (h_vis.groupby(["시간"])["값"].mean().sum()) if not h_vis.empty else np.nan
    # 가장 많은 시간대(5개 구간으로 묶음)
    time_bins = pd.IntervalIndex.from_tuples([(5,9),(9,12),(12,14),(14,18),(18,23)], closed='both')
    labels = ["05-09","09-12","12-14","14-18","18-23"]
    block = None
    if not h_vis.empty:
        hh = h_vis.groupby("시간")["값"].mean().reindex(range(24), fill_value=0).reset_index()
        hh["block"] = pd.cut(hh["시간"], bins=[5,9,12,14,18,23], labels=labels, include_lowest=True, right=True)
        bb = hh.groupby("block")["값"].sum().reset_index().dropna()
        if not bb.empty:
            block = bb.sort_values("값", ascending=False).iloc[0]["block"]
    return amt, daily_vis, str(block) if block else None

# maintain selection by click
if "selected_dong" not in st.session_state:
    st.session_state["selected_dong"] = sel_dongs[0] if sel_dongs else None

# --------------------
# Layout: Map + Summary
# --------------------
left, right = st.columns([1.3, 1])

with left:
    st.subheader("🗺 Choropleth (행정동)")
    df_map = map_df_for_metric(metric, sel_years, sel_dongs)
    if df_map.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    else:
        if BUSAN_GEOJSON and Path(BUSAN_GEOJSON).exists():
            gj = json.loads(Path(BUSAN_GEOJSON).read_text(encoding="utf-8"))
            fig = px.choropleth_mapbox(
                df_map, geojson=gj, featureidkey="properties.HADM_CD",  # GeoJSON 키에 맞게 수정 필요
                locations="행정동코드", color="value_norm",
                hover_name="행정동명", mapbox_style="carto-positron",
                center={"lat":35.18, "lon":129.07}, zoom=9, opacity=0.75,
                color_continuous_scale="Viridis"
            )
            fig.update_traces(marker_line_width=0.2, marker_line_color="white")
            st.plotly_chart(fig, use_container_width=True, key="map")

            # click selection
            ev = st.session_state.get("map", None)
            # Streamlit doesn't give clickData directly via plotly_chart; use Plotly events via st.session_state?
            # Workaround: provide a selectbox list synchronized with click helper.
        else:
            st.dataframe(df_map.sort_values("value", ascending=False).head(20))

    # As an alternative to click, provide a single-select dong picker synced with the map
    cand = df_map["행정동명"].unique().tolist() if not df_map.empty else []
    sel_one = st.selectbox("지도의 행정동 요약 보기(클릭 대안)", ["(선택)"] + cand, index=0)
    if sel_one != "(선택)":
        st.session_state["selected_dong"] = sel_one

with right:
    st.subheader("📌 요약 패널 (선택 행정동)")
    target_dong = st.session_state.get("selected_dong")
    if not target_dong:
        st.info("지도의 행정동을 클릭하거나, 왼쪽 아래의 선택박스에서 하나를 선택하세요.")
    else:
        amt, daily_vis, best_block = compute_kpis(target_dong, sel_years)

        k1, k2 = st.columns(2)
        with k1:
            st.metric("월 평균 이용금액 (원)", f"{amt:,.0f}" if pd.notnull(amt) else "-")
        with k2:
            st.metric("일 평균 방문인구 (명)", f"{daily_vis:,.0f}" if pd.notnull(daily_vis) else "-")
        if best_block:
            st.success(f"가장 많은 시간대: **{best_block}시**")

        # 성별 비교
        gg = g[(g["행정동명"]==target_dong) & (g["연도"].isin(sel_years)) & (g["지표"].isin(["visitors","residents","workers","trx_cnt","trx_amt"]))]
        if not gg.empty:
            fig2 = px.bar(gg.groupby(["지표","성별"])["값"].mean().reset_index(),
                          x="지표", y="값", color="성별", barmode="group", text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

        # 연령대
        aa = a[(a["행정동명"]==target_dong) & (a["연도"].isin(sel_years))]
        if not aa.empty:
            typ = st.radio("연령대 유형", ["amt","cnt"], horizontal=True, key="age_typ")
            bar = aa[aa["유형"]==typ].groupby(["연령대"])["값"].mean().reset_index()
            fig3 = px.bar(bar, x="연령대", y="값", text_auto=True)
            st.plotly_chart(fig3, use_container_width=True)

st.caption("※ 요일별 정보는 제공된 데이터에 포함되어 있지 않아 시간대 기반으로 대체 요약합니다.")
