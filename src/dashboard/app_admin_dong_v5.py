# streamlit run app_admin_dong_v5.py
# -*- coding: utf-8 -*-
"""
부산 행정동 생활·소비 대시보드 v5
요구 반영:
- 연도 → 행정동 토글(사이드바) 흐름
- 제공된 GeoJSON 사용, properties 키 직접 선택
- 지도 경계선(흰색) + 그라데이션
- 상세: 같은 레이아웃 공간에서
  * 인구(방문/거주/직장) 그래프
  * 이용건수/이용금액 그래프
  * 작은 성별 도넛
- 숫자/단위 변환 금지(원 데이터 그대로 사용)
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="부산 행정동 대시보드 v5", layout="wide")

st.title("📊 부산 행정동 생활·소비 대시보드 v5")
st.caption("연도 → 행정동 선택 · GeoJSON 속성키 지정 · 카테고리별 그래프 · 작은 성별 도넛 · 숫자 단위 그대로")

# --------- Sidebar: 경로/캐시 ----------
DATA_CSV = st.sidebar.text_input("CSV 경로", value="통합_행정동_데이터_1st.csv")
CACHE_DIR = st.sidebar.text_input("캐시 경로", value="./cache")
# 필요시 본인 경로로 교체
BUSAN_GEOJSON = st.sidebar.text_input("GeoJSON 경로", value="busan_205.geojson")

from dash_preprocess import build_cache
cache_dir = Path(CACHE_DIR)
meta_path = cache_dir / "meta.json"
if st.sidebar.button("🔄 캐시 생성/갱신"):
    info = build_cache(DATA_CSV, CACHE_DIR)
    st.sidebar.success(f"완료: {info}")
if meta_path.exists():
    import json as _json
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
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

# --------- 연도 → 행정동 선택 ----------
years = sorted(pd.concat([df["연도"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_years = st.sidebar.multiselect("① 연도 선택", years, default=years)
if not sel_years:
    st.info("좌측에서 먼저 **연도**를 선택하세요."); st.stop()

dongs = sorted(pd.concat([df["행정동명"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_dongs = st.sidebar.multiselect("② 행정동 선택 (선택 안 하면 전체)", dongs, default=[])

# --------- GeoJSON 로드/속성키 ----------
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
    st.sidebar.warning("GeoJSON이 없으면 지도는 표로 대체됩니다.")

# --------- Helper ----------
def map_frame(metric: str, source: str, years_sel, dongs_sel):
    """metric ∈ {'visitors','residents','workers','trx_cnt','trx_amt'}; source ∈ {'시간대','성별'}"""
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
    # ★ 금액 등 어떤 지표도 변환하지 않음 (요청사항)
    return df_map

def hourly_line(dong: str, metric: str, years_sel, height=250):
    data = h[(h["행정동명"]==dong) & (h["연도"].isin(years_sel)) & (h["지표"]==metric)]
    if data.empty:
        return go.Figure()
    s = data.groupby("시간")["값"].mean().reindex(range(24), fill_value=0).reset_index()
    fig = px.line(s, x="시간", y="값", markers=True)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(dtick=1), height=height)
    return fig

def donut_small(dong: str, metric: str, years_sel, height=180):
    data = g[(g["행정동명"]==dong) & (g["연도"].isin(years_sel)) & (g["지표"]==metric)]
    if data.empty:
        return go.Figure()
    dd = data.groupby("성별")["값"].mean().reset_index()
    fig = go.Figure(go.Pie(labels=dd["성별"], values=dd["값"], hole=0.6))
    fig.update_traces(textinfo="percent", hoverinfo="label+value", showlegend=False)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=height)
    return fig

# --------- 지도 ----------
st.markdown("### 🗺 행정동 Choropleth")
c1, c2, c3 = st.columns([1.1,1,1])
with c1:
    metric_map = st.selectbox("지도 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0)
with c2:
    source_map = st.selectbox("지도 집계 소스", ["시간대(24h 평균)", "성별(월 평균)"], index=0)
with c3:
    norm_on = st.checkbox("0-1 정규화", value=True)

mf = map_frame(metric_map, source_map, sel_years, sel_dongs)

if gj is not None and featureidkey:
    if mf.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
    else:
        mf["행정동코드"] = mf["행정동코드"].astype(str)
        color_s = mf["value"].astype(float)
        mf["__color__"] = (color_s - color_s.min()) / ((color_s.max() - color_s.min()) or 1.0) if norm_on else color_s
        fig = px.choropleth_mapbox(
            mf,
            geojson=gj,
            featureidkey=f"properties.{featureidkey}",
            locations="행정동코드",
            color="__color__",
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
    st.dataframe(mf.sort_values("value", ascending=False).head(20))

# --------- 같은 공간: 인구 3종 + 이용건수/이용금액 + 작은 도넛 ----------
st.markdown("## 📌 선택 행정동 상세")
cands = mf["행정동명"].unique().tolist() if not mf.empty else dongs
sel_one = st.selectbox("행정동 선택", ["(선택)"] + cands, index=0)
if sel_one == "(선택)":
    st.stop()

topL, topM, topR, topDonut = st.columns([1,1,1,0.6])
with topL:
    st.markdown("**🚶 유동인구(방문)**")
    st.plotly_chart(hourly_line(sel_one, "visitors", sel_years, height=220), use_container_width=True)
with topM:
    st.markdown("**🏠 거주인구**")
    st.plotly_chart(hourly_line(sel_one, "residents", sel_years, height=220), use_container_width=True)
with topR:
    st.markdown("**🧳 직장인구**")
    st.plotly_chart(hourly_line(sel_one, "workers", sel_years, height=220), use_container_width=True)
with topDonut:
    st.markdown("**🚻 성별(도넛)**")
    st.plotly_chart(donut_small(sel_one, "visitors", sel_years, height=160), use_container_width=True)

botL, botR = st.columns(2)
with botL:
    st.markdown("**🧾 이용건수**")
    st.plotly_chart(hourly_line(sel_one, "trx_cnt", sel_years, height=260), use_container_width=True)
with botR:
    st.markdown("**💴 이용금액**")
    st.plotly_chart(hourly_line(sel_one, "trx_amt", sel_years, height=260), use_container_width=True)

st.caption("※ 숫자 단위/스케일은 원 데이터 그대로를 사용합니다.")