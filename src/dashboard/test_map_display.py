# streamlit run app_admin_dong_v7.py
# -*- coding: utf-8 -*-
"""
부산 행정동 대시보드 v7
- 최종 수정: 지도 표시용 이름 형식 통일 ('구/군' 제거)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------
st.set_page_config(page_title="부산 행정동 대시보드 v7", layout="wide")
st.title("📊 부산 행정동 생활·소비 대시보드 v7")
st.caption("연도 → 지도(경계/그라데이션) → 행정동 정보 윈도우 · 상위5위 · 선택 그래프 · 숫자 스케일 변경 없음")

# ------------------------------------------------------
# 사이드바: 경로/캐시/연도/옵션
# ------------------------------------------------------
DATA_CSV = st.sidebar.text_input("CSV 경로", value="통합_행정동_데이터_1st.csv")
CACHE_DIR = st.sidebar.text_input("캐시 경로", value="./cache")
BUSAN_GEOJSON = st.sidebar.text_input("GeoJSON 경로", value="busan_205.geojson")

from dash_preprocess import build_cache
if st.sidebar.button("🔄 캐시 생성/갱신"):
    info = build_cache(DATA_CSV, CACHE_DIR)
    st.sidebar.success(f"완료: {info}")

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

# 연도 토글
years = sorted(pd.concat([df["연도"] for df in [g,h,a] if not df.empty]).unique().tolist())
sel_year = st.sidebar.radio("① 연도 선택(토글·단일)", options=years, index=len(years)-1 if years else 0)

# GeoJSON 로드 & 속성키
gj, featureidkey = None, None
feature_keys = []
if BUSAN_GEOJSON and Path(BUSAN_GEOJSON).exists():
    gj = json.loads(Path(BUSAN_GEOJSON).read_text(encoding="utf-8"))
    if gj.get("features"):
        props = list(gj["features"][0].get("properties", {}).keys())
        guesses = ["ADM_NM", "ADM_CD","EMD_CD","HADM_CD","ADM_DR_CD","adm_cd","code","행정동코드"]
        in_common = [k for k in guesses if k in props]
        feature_keys = in_common + [k for k in props if k not in in_common]
    default_index = 0
    if "ADM_NM" in feature_keys:
        default_index = feature_keys.index("ADM_NM")
    elif "ADM_CD" in feature_keys:
        default_index = feature_keys.index("ADM_CD")
    featureidkey = st.sidebar.selectbox("GeoJSON의 동코드/동이름 속성 선택", feature_keys, index=default_index)
else:
    st.sidebar.warning("GeoJSON이 없으면 지도는 표로 대체됩니다.")

# ------------------------------------------------------
# 컬러 팔레트
# ------------------------------------------------------
COLOR_VISITORS = "#2ca02c"
COLOR_RESIDENTS = "#9467bd"
COLOR_WORKERS  = "#8c564b"
COLOR_TRX_CNT  = "#ff7f0e"
COLOR_TRX_AMT  = "#17becf"
COLOR_MALE     = "#1f77b4"
COLOR_FEMALE   = "#d62728"

# ------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------
def _filter_g(year: int, metric: str, dong: str|None=None):
    q = g[(g["연도"]==year) & (g["지표"]==metric)]
    if dong and dong != "(선택)":
        q = q[q["행정동명"]==dong]
    return q

def _filter_h(year: int, metric: str, dong: str|None=None):
    q = h[(h["연도"]==year) & (h["지표"]==metric)]
    if dong and dong != "(선택)":
        q = q[q["행정동명"]==dong]
    return q

def df_map_cont(metric: str, year: int) -> pd.DataFrame:
    q = _filter_h(year, metric)
    if q.empty:
        return pd.DataFrame(columns=["행정동명","value"])
    df = q.groupby("행정동명")["값"].mean().reset_index(name="value")
    return df[["행정동명","value"]]

def df_map_gender(metric: str, year: int) -> pd.DataFrame:
    q = _filter_g(year, metric)
    if q.empty:
        return pd.DataFrame(columns=["행정동명","diff","male","female"])
    p = q.pivot_table(index="행정동명", columns="성별", values="값", aggfunc="mean").reset_index()
    for c in ["male","female"]:
        if c not in p.columns: p[c] = np.nan
    p["diff"] = p["male"].fillna(0) - p["female"].fillna(0)
    return p[["행정동명","diff","male","female"]]

def rank_percentile(df_map: pd.DataFrame, value_col: str, dong_name: str):
    if df_map.empty or dong_name not in df_map["행정동명"].values:
        return None, None
    t = df_map[["행정동명", value_col]].dropna().sort_values(value_col, ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t)+1)
    row = t[t["행정동명"]==dong_name].iloc[0]
    r = int(row["rank"])
    pct = 100.0 if len(t)==1 else 100.0*(1.0 - (r-1)/(len(t)-1))
    return r, pct

def hourly_line(dong: str, metric: str, year: int, color: str, height=240):
    d = _filter_h(year, metric, dong)
    if d.empty:
        return go.Figure(), None
    s = d.groupby("시간")["값"].mean().reindex(range(24), fill_value=0).reset_index()
    fig = px.line(s, x="시간", y="값", markers=True)
    fig.update_traces(line_color=color)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(dtick=1), height=height)
    peak = int(s.loc[s["값"].idxmax(), "시간"])
    fig.add_vline(x=peak, line_dash="dot")
    return fig, peak

def donut_gender(dong: str, metric: str, year: int, height=160):
    d = _filter_g(year, metric, dong)
    if d.empty:
        return go.Figure()
    dd = d.groupby("성별")["값"].mean().reset_index()
    colors = [COLOR_MALE if s=="male" else COLOR_FEMALE for s in dd["성별"]]
    fig = go.Figure(go.Pie(labels=dd["성별"], values=dd["값"], hole=0.6))
    fig.update_traces(textinfo="percent", hoverinfo="label+value", showlegend=False, marker=dict(colors=colors))
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=height)
    return fig

# ------------------------------------------------------
# 상단: 상위 5위 카드 + 지도
# ------------------------------------------------------
st.markdown("### 🏆 선택 연도의 상위 5위 행정동")
top_metric = st.radio("상위 5위 기준 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], horizontal=True, index=0)
df_top = df_map_cont(top_metric, sel_year)
if df_top.empty:
    st.info("상위5위: 해당 연도의 데이터가 없습니다.")
else:
    top5 = df_top.sort_values("value", ascending=False).head(5)
    cols = st.columns(5)
    for i,(idx,row) in enumerate(top5.iterrows()):
        with cols[i]:
            st.metric(f"{i+1}위", row["행정동명"], delta=None)

st.markdown("### 🗺 전체 지도")
map_left, map_right = st.columns([1.2, 1])

with map_left:
    map_mode = st.radio("지도 유형", ["지표(연속형)", "성별(남/여 비교)"], horizontal=True)
    if map_mode == "지표(연속형)":
        metric_map = st.selectbox("지표 선택", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="metric_map")
        df_m = df_map_cont(metric_map, sel_year)
        if gj is not None and featureidkey:
            if df_m.empty:
                st.warning("지표 데이터가 없습니다.")
            else:
                # ==============================================================================
                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                # 지도 시각화용 데이터프레임 복사 및 이름 형식 변경
                df_for_map = df_m.copy()
                # '남구 감만1동' -> '감만1동'으로 변경. 공백이 없는 이름은 그대로 유지.
                df_for_map['행정동명_지도용'] = df_for_map['행정동명'].apply(lambda x: x.split()[-1])
                
                fig = px.choropleth_mapbox(
                    df_for_map,  # 수정된 데이터프레임 사용
                    geojson=gj,
                    featureidkey=f"properties.{featureidkey}",
                    locations="행정동명_지도용",  # '구/군'이 제거된 이름으로 매칭
                    color="value",
                    hover_name="행정동명",  # 마우스를 올렸을 때는 원래 전체 이름 표시
                    mapbox_style="carto-positron",
                    center={"lat":35.18, "lon":129.07}, zoom=9, opacity=0.85,
                    color_continuous_scale="Viridis"
                )
                # ==============================================================================
                fig.update_traces(marker_line_width=1.2, marker_line_color="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df_m.sort_values("value", ascending=False).head(20))
    else:
        metric_g = st.selectbox("성별 비교 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="metric_g")
        df_m = df_map_gender(metric_g, sel_year)
        if gj is not None and featureidkey:
            if df_m.empty:
                st.warning("지표 데이터가 없습니다.")
            else:
                # ==============================================================================
                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                df_for_map = df_m.copy()
                df_for_map['행정동명_지도용'] = df_for_map['행정동명'].apply(lambda x: x.split()[-1])
                
                fig = px.choropleth_mapbox(
                    df_for_map, # 수정된 데이터프레임 사용
                    geojson=gj,
                    featureidkey=f"properties.{featureidkey}",
                    locations="행정동명_지도용", # '구/군'이 제거된 이름으로 매칭
                    color="diff",
                    hover_name="행정동명", # 마우스를 올렸을 때는 원래 전체 이름 표시
                    mapbox_style="carto-positron",
                    center={"lat":35.18, "lon":129.07}, zoom=9, opacity=0.85,
                    color_continuous_scale="RdBu", color_continuous_midpoint=0.0
                )
                # ==============================================================================
                fig.update_traces(marker_line_width=1.2, marker_line_color="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df_m.sort_values("diff", ascending=False).head(20))

# ------------------------------------------------------
# 우측: 정보 윈도우 (행정동 선택 → 요약/그래프)
# ------------------------------------------------------
with map_right:
    st.subheader("📌 정보 윈도우")
    # 드롭다운 목록은 전체 이름('남구 감만1동')을 그대로 사용
    all_dongs = sorted(g["행정동명"].unique().tolist())
    sel_dong = st.selectbox("행정동 선택", options=["(선택)"] + all_dongs, index=0)

    if sel_dong and sel_dong != "(선택)":
        if map_mode == "지표(연속형)":
            base_df = df_map_cont(metric_map, sel_year); base_col = "value"
        else:
            base_df = df_map_gender(metric_g, sel_year).rename(columns={"diff":"value"}); base_col = "value"
        # 순위 계산 시에도 전체 이름으로 된 base_df 사용
        rnk, pct = rank_percentile(base_df, base_col, sel_dong)

        st.markdown(f"- **선택 연도:** {sel_year}")
        st.markdown(f"- **선택 행정동:** {sel_dong}")
        if rnk is not None:
            st.success(f"지도 기준 지표 순위: **{rnk}위** (상위 **{pct:.1f}%**)")

        badges = []
        for m, label in [("visitors","유동"),("residents","거주"),("workers","직장"),("trx_cnt","건수"),("trx_amt","금액")]:
            base = df_map_cont(m, sel_year)
            rr, pp = rank_percentile(base, "value", sel_dong)
            if rr is None:
                continue
            if pp >= 90: badges.append(f"**{label} 상위권(상위 {pp:.0f}%)**")
            elif pp <= 10: badges.append(f"**{label} 하위권(하위 {100-pp:.0f}%)**")
        if badges:
            st.markdown("🔎 **특징:** " + " · ".join(badges))

        st.markdown("**🚻 성별 비중 (기준: 월평균 값 비중)**")
        donut_metric = st.selectbox("도넛 지표", ["visitors","residents","workers","trx_cnt","trx_amt"], index=0, key="donut")
        st.plotly_chart(donut_gender(sel_dong, donut_metric, sel_year, height=150), use_container_width=True)

        st.markdown("### 📈 선택 그래프 (체크한 것만 표시)")
        c_vis = st.checkbox("유동인구(방문) 시간대", value=False)
        c_res = st.checkbox("거주인구 시간대", value=False)
        c_wrk = st.checkbox("직장인구 시간대", value=False)
        c_cnt = st.checkbox("이용건수 시간대", value=False)
        c_amt = st.checkbox("이용금액 시간대", value=False)

        g1, g2 = st.columns(2)
        if c_vis:
            with g1:
                fig, peak = hourly_line(sel_dong, "visitors", sel_year, COLOR_VISITORS, 260)
                st.plotly_chart(fig, use_container_width=True)
                if peak is not None: st.caption(f"히트 타임: **{peak}시**")
        if c_res:
            with g2:
                fig, peak = hourly_line(sel_dong, "residents", sel_year, COLOR_RESIDENTS, 260)
                st.plotly_chart(fig, use_container_width=True)
                if peak is not None: st.caption(f"히트 타임: **{peak}시**")
        if c_wrk:
            with g1:
                fig, peak = hourly_line(sel_dong, "workers", sel_year, COLOR_WORKERS, 260)
                st.plotly_chart(fig, use_container_width=True)
                if peak is not None: st.caption(f"히트 타임: **{peak}시**")
        if c_cnt:
            with g2:
                fig, peak = hourly_line(sel_dong, "trx_cnt", sel_year, COLOR_TRX_CNT, 260)
                st.plotly_chart(fig, use_container_width=True)
                if peak is not None: st.caption(f"히트 타임: **{peak}시**")
        if c_amt:
            with g1:
                fig, peak = hourly_line(sel_dong, "trx_amt", sel_year, COLOR_TRX_AMT, 260)
                st.plotly_chart(fig, use_container_width=True)
                if peak is not None: st.caption(f"히트 타임: **{peak}시**")
    else:
        st.info("행정동을 선택하면 요약과 그래프가 표시됩니다.")
