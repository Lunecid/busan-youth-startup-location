# streamlit run app_final.py
# -*- coding: utf-8 -*-
"""
부산 행정동 대시보드 v17.4
- 라이트 테마 고정(차트/텍스트/셀렉트박스 색상 커스텀)
- 지도는 항상 표시, 우측 패널만 '분포·추이·비중' 토글
- 연도/월 동기화: 월을 선택하면 모든 계산/제목/TOP5가 해당 월 기준, "(전체)"면 연평균
- TOP5: 1위 강조, 내부 라벨, 상위가 맨 위
- 상위 퍼센트 정의 수정: 상위 (rank/total*100)%  → 1위 ≈ 상위 0~1%
- CSV(와이드) 자동 전처리: '기준년월' → 연도/월 파싱, gender/hourly/age 캐시 생성
- KPI 카드: 숫자/문자열 모두 안전 처리(예: '최다 시간대' 같은 문자열)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ===================== Theme =====================
st.set_page_config(page_title="부산 행정동 인사이트", layout="wide", initial_sidebar_state="expanded")

THEME = {
    "bg": "#F7FAFC",          # 페이지 배경
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#0F172A",        # 기본 글자
    "muted": "#6B7280",
    "primary": "#7DD3FC",     # 기본 막대/선
    "accent": "#2DD4BF",      # 1위 강조
    "good": "#10B981",
    "warn": "#F59E0B",
    "grid": "#EDF2F7"
}

st.markdown(f"""
<style>
/* 전체 배경 */
.stApp, .block-container {{ background: {THEME["bg"]}!important; }}

/* 카드 */
.card {{
  background:{THEME["card"]}; border:1px solid {THEME["border"]};
  border-radius:18px; padding:16px 18px; box-shadow:0 4px 18px rgba(2,12,27,.06);
}}

/* KPI */
.kpi {{ display:flex; gap:12px; align-items:flex-start; justify-content:space-between; }}
.kpi .title {{ color:{THEME["muted"]}; font-size:13px; }}
.kpi .value {{ font-size:28px; font-weight:800; color:{THEME["text"]}; }}

/* 알림 */
.note {{ border-left:6px solid {THEME["good"]}; background:#F0FDF4; padding:12px 14px; border-radius:12px; color:#065f46; }}

/* 사이드바 선택박스/라디오 - 라이트 톤으로 */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
  background: #FFFFFF; color:{THEME["text"]};
  border:1px solid {THEME["border"]}; border-radius:10px;
}}
section[data-testid="stSidebar"] .stRadio label span {{
  color:{THEME["text"]} !important;
}}
</style>
""", unsafe_allow_html=True)

st.title("✨ 부산 행정동 생활·소비 인사이트 대시보드")

# ===================== 매핑/상수 =====================
METRIC_MAP = {
    "visitors": "방문인구",
    "residents": "거주인구",
    "workers": "직장인구",
    "trx_cnt": "소비 건수",
    "trx_amt": "소비 금액",
}
METRICS_ORDER = list(METRIC_MAP.keys())

AGE_LABELS = {
    "under20": "20세 미만", "20s": "20대", "30s": "30대", "40s": "40대",
    "50s": "50대", "60s": "60대", "over70": "70세 이상",
}
AGE_ORDER = ["under20","20s","30s","40s","50s","60s","over70"]
AGE_ORDER_MAP = {k:i for i,k in enumerate(AGE_ORDER)}

# ===================== 유틸 =====================
def parse_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """기준년월 → 연도/월. 이미 있으면 형식 정리."""
    if "연도" in df.columns and "월" in df.columns:
        df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
        df["월"] = pd.to_numeric(df["월"], errors="coerce").astype("Int64")
        return df
    if "기준년월" not in df.columns:
        raise ValueError("CSV에 '연도/월'이 없고 '기준년월'도 없습니다.")
    base = df["기준년월"].astype(str).str.strip()
    base = base.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    is_yyyymm = base.str.fullmatch(r"\d{6}", na=False)
    base = np.where(is_yyyymm, base.str[:4] + "-" + base.str[4:6], base)
    dt = pd.to_datetime(pd.Series(base) + "-01", errors="coerce")
    keep = ~dt.isna()
    df = df.loc[keep].copy(); dt = dt.loc[keep]
    df["연도"] = dt.dt.year.astype("Int64")
    df["월"] = dt.dt.month.astype("Int64")
    return df

def build_cache(csv_path: str, cache_dir: str):
    """와이드 CSV → 롱 3종 parquet."""
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8").copy()
    df = parse_year_month(df)

    # 1) 성별 요약
    metric_kr2en = {
        "평균방문인구수": "visitors",
        "평균주거인구수": "residents",
        "평균직장인구수": "workers",
        "평균이용건수": "trx_cnt",
        "평균이용금액": "trx_amt",
    }
    rows_g=[]
    for kr,en in metric_kr2en.items():
        for sex in ["male","female"]:
            col=f"{kr}_{sex}"
            if col in df.columns:
                sub=df[["행정동명","연도","월"]].copy()
                sub["지표"]=en; sub["성별"]=sex
                sub["값"]=pd.to_numeric(df[col], errors="coerce")
                rows_g.append(sub)
    g = pd.concat(rows_g, ignore_index=True) if rows_g else pd.DataFrame(columns=["행정동명","연도","월","지표","성별","값"])
    g.to_parquet(cache/"gender.parquet", index=False)

    # 2) 시간대
    families={"방문인구":"visitors","주거인구":"residents","직장인구":"workers","이용건수":"trx_cnt","이용금액":"trx_amt"}
    rows_h=[]
    for kr,en in families.items():
        hour_cols=[c for c in df.columns if isinstance(c,str) and c.startswith(f"{kr}_")]
        for hc in hour_cols:
            hh=hc.split("_")[-1]
            try: h=int(hh)
            except: continue
            sub=df[["행정동명","연도","월"]].copy()
            sub["지표"]=en; sub["시간"]=h
            sub["값"]=pd.to_numeric(df[hc], errors="coerce")
            rows_h.append(sub)
    h=pd.concat(rows_h, ignore_index=True) if rows_h else pd.DataFrame(columns=["행정동명","연도","월","지표","시간","값"])
    h.to_parquet(cache/"hourly.parquet", index=False)

    # 3) 연령
    age_cols=[c for c in df.columns if isinstance(c,str) and (c.startswith("amt_") or c.startswith("cnt_"))]
    rows_a=[]
    for col in age_cols:
        sub=df[["행정동명","연도","월"]].copy()
        sub["지표"]=col; sub["값"]=pd.to_numeric(df[col], errors="coerce")
        rows_a.append(sub)
    a=pd.concat(rows_a, ignore_index=True) if rows_a else pd.DataFrame(columns=["행정동명","연도","월","지표","값"])
    a.to_parquet(cache/"age.parquet", index=False)

def scope_filter(df: pd.DataFrame, year: int, month_sel):
    """연-월 필터 및 타이틀 텍스트 반환."""
    if month_sel == "(전체)":
        df2 = df[df["연도"]==year].copy()
        scope_text = f"{year}년 연평균"
    else:
        df2 = df[(df["연도"]==year) & (df["월"]==month_sel)].copy()
        scope_text = f"{year}년 {month_sel}월"
    return df2, scope_text

def get_top_percent(rank:int, total:int)->str:
    if total<=0: return ""
    return f"상위 {rank/total*100:.1f}%"

def get_dong_rank(df_grouped: pd.DataFrame, dong_name: str):
    if df_grouped.empty or "행정동명" not in df_grouped.columns or "값" not in df_grouped.columns:
        return None, None
    if dong_name not in df_grouped["행정동명"].values:
        return None, None
    sorted_df = df_grouped.sort_values("값", ascending=False).reset_index(drop=True)
    row = sorted_df[sorted_df["행정동명"] == dong_name]
    if row.empty: return None, None
    rank = int(row.index[0]) + 1
    total = len(sorted_df)
    return f"{rank}위 / {total}곳", get_top_percent(rank, total)

def mean_by_dong(df: pd.DataFrame)->pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["행정동명","값"])
    return df.groupby("행정동명", as_index=False)["값"].mean()

def apply_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=THEME["card"],
        plot_bgcolor=THEME["card"],
        font=dict(color=THEME["text"]),
        xaxis=dict(gridcolor=THEME["grid"]),
        yaxis=dict(gridcolor=THEME["grid"]),
        margin=dict(l=10, r=16, t=34, b=10),
    )
    return fig

def kpi_card(title: str, value, delta: float | None = None):
    if value is None:
        val = "–"
    elif isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
        val = f"{value:,.0f}"
    else:
        val = str(value)

    delta_html = ""
    if isinstance(delta, (int, float)):
        arrow = "↑" if delta >= 0 else "↓"
        color = THEME["good"] if delta >= 0 else THEME["warn"]
        delta_html = f'<div style="color:{color}; font-weight:700;">{arrow} {abs(delta):.1f}%</div>'

    st.markdown(f'''
    <div class="card kpi">
      <div>
        <div class="title">{title}</div>
        <div class="value">{val}</div>
      </div>
      {delta_html}
    </div>
    ''', unsafe_allow_html=True)

def render_top5_bar(top5_df: pd.DataFrame, title: str):
    if top5_df.empty:
        return None
    ylabels = [str(s).split()[-1] for s in top5_df["행정동명"]]
    colors = [THEME["accent"]] + [THEME["primary"]] * (len(top5_df)-1)
    fig = go.Figure(go.Bar(
        x=top5_df["값"], y=ylabels, orientation="h",
        marker=dict(color=colors, line=dict(color=THEME["border"], width=0.8)),
        text=[f"{v:,.0f}" for v in top5_df["값"]],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#FFFFFF", size=12),
        hovertemplate="%{y} • %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, showlegend=False, uniformtext_minsize=12, uniformtext_mode="hide",
        xaxis=dict(title="", automargin=True, showgrid=True), yaxis=dict(title=""), height=260
    )
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig)

def prepare_age_df(a: pd.DataFrame, sel_year: int, sel_dong: str, month_sel):
    base,_ = scope_filter(a[(a["행정동명"]==sel_dong)], sel_year, month_sel)
    if base.empty: return pd.DataFrame(columns=["kind","age_key","연령대","order","값"])
    agg = base.groupby("지표", as_index=False)["값"].mean()
    agg["kind"] = agg["지표"].str.extract(r"^(amt|cnt)", expand=False)
    agg["age_key"] = agg["지표"].str.replace(r"^(amt|cnt)_","", regex=True)
    agg["연령대"] = agg["age_key"].map(AGE_LABELS).fillna(agg["age_key"])
    agg["order"] = agg["age_key"].map(AGE_ORDER_MAP).fillna(999).astype(int)
    return agg[["kind","age_key","연령대","order","값"]].dropna(subset=["kind"])

# ===================== 사이드바 / 데이터 로딩 =====================
with st.sidebar:
    st.header("⚙️ 컨트롤")
    DATA_CSV = st.text_input("CSV 경로", value="통합_행정동_데이터_1st.csv")
    CACHE_DIR = st.text_input("캐시 경로", value="./cache")
    GEOJSON_PATH = st.text_input("GeoJSON 경로", value="busan_205.geojson")

    if st.button("🔄 캐시 생성/갱신"):
        with st.spinner("데이터 전처리 중..."):
            build_cache(DATA_CSV, CACHE_DIR)
        st.success("캐시 생성 완료!")

    @st.cache_data(show_spinner="데이터 로딩 중...")
    def load_data(cache_dir: str):
        c = Path(cache_dir)
        g = pd.read_parquet(c/"gender.parquet") if (c/"gender.parquet").exists() else pd.DataFrame()
        h = pd.read_parquet(c/"hourly.parquet") if (c/"hourly.parquet").exists() else pd.DataFrame()
        a = pd.read_parquet(c/"age.parquet") if (c/"age.parquet").exists() else pd.DataFrame()
        return g,h,a
    g,h,a = load_data(CACHE_DIR)

    if g.empty or h.empty or a.empty:
        st.warning("캐시가 비어있어요. CSV 경로 확인 후 ‘캐시 생성/갱신’을 눌러주세요.")
        st.stop()

    years = sorted(h["연도"].dropna().unique(), reverse=True)
    sel_year = st.selectbox("조회 연도", years)

    months = sorted(h.loc[h["연도"]==sel_year,"월"].dropna().unique())
    sel_month = st.selectbox("월 선택 (연평균은 '(전체)')", ["(전체)"]+list(months), index=0)

    gj = None
    if GEOJSON_PATH and Path(GEOJSON_PATH).exists():
        gj = json.loads(Path(GEOJSON_PATH).read_text(encoding="utf-8"))

# ===================== TOP5 (연/월 동기화) =====================
scope_df, scope_text = scope_filter(h, sel_year, sel_month)
st.markdown(f"#### 🏆 {scope_text} 지표별 TOP5")

cols = st.columns(len(METRICS_ORDER))
for i, metric_eng in enumerate(METRICS_ORDER):
    metric_kor = METRIC_MAP[metric_eng]
    with cols[i]:
        dat = scope_df[scope_df["지표"]==metric_eng]
        if dat.empty:
            st.caption("데이터 없음")
        else:
            top5 = dat.groupby("행정동명")["값"].mean().nlargest(5).reset_index()
            fig = render_top5_bar(top5, f"{scope_text} {metric_kor} TOP5")
            if fig is None: st.caption("데이터 없음")
            else: st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===================== 지도 & 오른쪽 패널 (지도 항상 표시) =====================
map_col, panel_col = st.columns([0.68, 0.32])

with map_col:
    st.markdown('<div class="card" style="padding:12px 14px; margin-bottom:8px;">', unsafe_allow_html=True)
    ca, cb = st.columns([0.55, 0.45])
    with ca:
        view_mode = st.radio("패널 보기모드", ["분포(요약)","추이","비중"], horizontal=True, index=0)
    with cb:
        metric_map_kor = st.selectbox("지표", list(METRIC_MAP.values()), index=4)
    st.markdown('</div>', unsafe_allow_html=True)

    # 지도는 항상 표시 (선택 지표 + 연/월 스코프)
    st.markdown("#### 🗺️ 행정동별 분포 지도")
    metric_map_eng = [k for k,v in METRIC_MAP.items() if v==metric_map_kor][0]
    map_src = scope_df[scope_df["지표"]==metric_map_eng][["행정동명","값"]]
    df_map = mean_by_dong(map_src)
    if gj and not df_map.empty:
        df_map["행정동명_지도용"] = df_map["행정동명"].astype(str).apply(lambda x: x.split()[-1])
        fig = px.choropleth_mapbox(
            df_map, geojson=gj, featureidkey="properties.ADM_NM",
            locations="행정동명_지도용", color="값", hover_name="행정동명",
            mapbox_style="carto-positron", center={"lat":35.18,"lon":129.07}, zoom=9.5,
            opacity=0.82, color_continuous_scale="Tealgrn"
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    elif not gj:
        st.info("GeoJSON 경로를 지정하면 지도가 표시됩니다.")
    else:
        st.caption("지도 데이터 없음")

with panel_col:
    st.markdown("#### 📎 상세 패널")
    all_dongs = sorted(h["행정동명"].dropna().astype(str).unique().tolist())
    sel_dong = st.selectbox("행정동", ["(선택)"]+all_dongs, index=0)

    if sel_dong == "(선택)":
        st.info("행정동을 선택하면 상세 정보가 표시됩니다.")
    else:
        # 공통 소스(연/월 스코프)
        metric_map_eng = [k for k,v in METRIC_MAP.items() if v==metric_map_kor][0]
        scope_dong = scope_df[scope_df["행정동명"]==sel_dong]

        if view_mode=="분포(요약)":
            # KPI: 소비금액/방문인구/최다시간대 (스코프 기준)
            amt = scope_dong[scope_dong["지표"]=="trx_amt"]["값"].mean()
            vis = scope_dong[scope_dong["지표"]=="visitors"]
            kpi_card(f"{scope_text} 소비 금액(평균)", None if np.isnan(amt) else amt)

            vmean = scope_dong[scope_dong["지표"]=="visitors"]["값"].mean()
            kpi_card(f"{scope_text} 방문인구(평균)", None if np.isnan(vmean) else vmean)

            peak = None
            if not vis.empty and vis["시간"].notna().any():
                peak = vis.groupby("시간")["값"].mean().idxmax()
            kpi_card("최다 시간대", f"{str(int(peak)).zfill(2)}시" if peak is not None else None)

            # 상위권 뱃지
            badges=[]
            for eng, kor in METRIC_MAP.items():
                d = scope_df[scope_df["지표"]==eng][["행정동명","값"]]
                if d.empty: continue
                r, p = get_dong_rank(mean_by_dong(d), sel_dong)
                if r:
                    try:
                        pnum=float(p.replace("상위","").replace("%","").strip())
                        if pnum<=10: badges.append(f"{kor} {p}")
                    except: pass
            if badges:
                st.markdown('<div class="note">🌟 <b>분석결과:</b> ' + ", ".join(badges) + ' 지표가 두드러집니다.</div>', unsafe_allow_html=True)

        elif view_mode=="추이":
            st.markdown(f"**{sel_dong} - {metric_map_kor} 추이 ({scope_text})**")
            d = scope_dong[scope_dong["지표"]==metric_map_eng]
            if d.empty:
                st.caption("데이터 없음")
            else:
                if d["시간"].notna().any():
                    s = d.groupby("시간")["값"].mean().reindex(range(24), fill_value=0)
                    fig = px.area(s, labels={"index":"시간","value":"평균값"})
                    st.plotly_chart(apply_theme(fig), use_container_width=True)
                else:
                    # 시간대가 없다면 월변화(해당 스코프가 '(전체)'일 때만 의미 있음)
                    s = d.groupby("월")["값"].mean().reset_index()
                    if not s.empty:
                        fig = px.line(s, x="월", y="값", markers=True)
                        st.plotly_chart(apply_theme(fig), use_container_width=True)
                    else:
                        st.caption("시간/월 차트용 데이터가 없습니다.")

        elif view_mode=="비중":
            st.markdown(f"**{sel_dong} - {metric_map_kor} 비중 ({scope_text})**")
            # 성별 비중 (스코프 적용)
            g_scope,_ = scope_filter(g[(g["행정동명"]==sel_dong)&(g["지표"]==metric_map_eng)], sel_year, sel_month)
            if g_scope.empty:
                g_scope,_ = scope_filter(g[(g["행정동명"]==sel_dong)&(g["지표"]=="trx_amt")], sel_year, sel_month)
            if not g_scope.empty:
                pie = g_scope.groupby("성별")["값"].mean().reset_index()
                fig = px.pie(pie, names="성별", values="값", title="성별 비중",
                             color_discrete_sequence=[THEME["primary"], THEME["accent"]])
                st.plotly_chart(apply_theme(fig), use_container_width=True)

            st.markdown("---")
            df_age = prepare_age_df(a, sel_year, sel_dong, sel_month)
            if not df_age.empty:
                cnt_df = df_age[df_age["kind"]=="cnt"].sort_values("order")
                amt_df = df_age[df_age["kind"]=="amt"].sort_values("order")
                if not cnt_df.empty:
                    fig_cnt = px.bar(cnt_df, x="연령대", y="값", title="연령별 매출수 (cnt_*)")
                    st.plotly_chart(apply_theme(fig_cnt), use_container_width=True)
                if not amt_df.empty:
                    fig_amt = px.bar(amt_df, x="연령대", y="값", title="연령별 매출금액 (amt_*)")
                    st.plotly_chart(apply_theme(fig_amt), use_container_width=True)
