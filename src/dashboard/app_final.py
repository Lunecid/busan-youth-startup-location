# app_recommendation_v28_grid_fixed.py
# -*- coding: utf-8 -*-
# 실행: streamlit run app_recommendation_v28_grid_fixed.py

import re, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================= Page / Theme =============================
st.set_page_config(page_title="청년 창업지 추천 대시보드 (Grid Fixed)", layout="wide")

# ---- Busan BI palette (근사 HEX) ----
BUSAN = {
    "light_blue": "#3DA5FF",    # Busan Global Light blue (C90 M30 근사)
    "orig_blue":  "#0033A0",    # Busan Original Blue (C100 M80 근사)
    "violet":     "#6C2BD9",    # Busan Open Violet (C75 M100 근사)
    "magenta":    "#E6007E",    # Busan Dynamic Magenta (M100 근사)
}

THEME = {
    "bg":"#FFFFFF", "card":"#FFFFFF", "border":"#E5E7EB",
    "text":"#0B1220", "muted":"#334155",
    "primary": BUSAN["orig_blue"],   # 강조 막대/선/일반 지표 텍스트
    "accent":  BUSAN["violet"],      # 행정동 이름, 섹션 라벨 포인트
    "grid":"#EEF2F7",
    "pale": BUSAN["light_blue"],     # 일반 막대
    "magenta": BUSAN["magenta"]      # 텍스트 하이라이트/여성
}

# ============================= Global CSS =============================
st.markdown(f"""
<style>
/* 레이아웃 & 기본 글자 */
.stApp, .block-container {{ background:{THEME["bg"]}!important; color:{THEME["text"]}; }}
.block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }}

.h-section {{ font-weight: 900; font-size: 1.5rem; margin: .25rem 0 .9rem;
  padding-left: .75rem; border-left: 6px solid {THEME["accent"]}; }}
.h-sub {{ font-weight: 800; font-size: 1.12rem; margin: .4rem 0 .6rem; color:{THEME["primary"]}; }}

/* 사이드바 */
[data-testid="stSidebar"] {{ background:{THEME["bg"]}!important; border-right:1px solid {THEME["border"]}; }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] .stTextInput>div>div,
[data-testid="stSidebar"] div[data-baseweb="select"]>div {{
  background:{THEME["card"]}!important; color:{THEME["text"]}!important;
  border:1px solid {THEME["border"]}!important; border-radius:10px!important;
}}
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSelectbox label p,
[data-testid="stSidebar"] .stTextInput label p {{ color:{THEME["text"]} !important; font-weight:800 !important; }}
[data-testid="stSidebar"] ::placeholder {{ color:#6B7280 !important; opacity:1; }}

/* 버튼 전역 */
.stButton > button, .stButton button,
button[kind="secondary"], [data-testid="baseButton-secondary"] {{
  background:#FFFFFF !important; color:{THEME["text"]}!important;
  border:1px solid {THEME["border"]}!important; border-radius:10px!important;
}}
/* 사이드바의 ‘캐시 생성/갱신’만 파란색 */
[data-testid="stSidebar"] div.stButton>button {{
  background:{THEME["primary"]}!important; color:#fff!important; border:0!important; border-radius:10px!important;
}}

/* Select / Popover 라이트 강제 */
:root {{ --menu-bg:#FFFFFF; --menu-txt:{THEME["text"]}; --menu-sel:#DBEAFE; }}
div[data-baseweb="popover"] > div, div[data-baseweb="menu"], div[role="listbox"] {{
  background: var(--menu-bg) !important; color: var(--menu-txt) !important;
  border: 1px solid {THEME["border"]} !important;
}}
div[role="listbox"] [role="option"] {{ background: var(--menu-bg) !important; color: var(--menu-txt) !important; }}
div[role="listbox"] [role="option"]:hover {{ background:#EFF6FF !important; color:{THEME["primary"]} !important; }}
div[role="listbox"] [role="option"][aria-selected="true"] {{
  background: var(--menu-sel) !important; color:{THEME["primary"]} !important; font-weight:800 !important;
}}

/* 카드 & 순위 리스트 */
.card {{ background:{THEME["card"]}; border:1px solid {THEME["border"]}; border-radius:14px;
  padding:14px; box-shadow:0 6px 16px rgba(15,23,42,.05); }}
.top5-card {{ height:248px; overflow:auto; }}
.top5-card h5 {{ font-size:1.0rem; color:{THEME["primary"]}; margin:0 0 8px 0; }}
ol.rank-list {{ list-style:none; padding-left:0; margin:0; }}
ol.rank-list li {{ display:flex; align-items:center; gap:.5rem; margin:.42rem 0; font-size:.97rem; }}
.rank-num {{ width:28px; text-align:center; font-weight:900; color:{THEME["accent"]}; }}
.rank-name {{ flex:1; }}
.rank-val {{ color:{THEME["muted"]}; font-size:.9rem; }}
.top3-val {{ color:{THEME["magenta"]}; font-weight:900; }}

/* KPI */
.kpi-title {{ color:{THEME["muted"]}; font-size:.95rem; font-weight:800; margin-bottom:.25rem; }}
.kpi-value {{ color:{THEME["primary"]}; font-size:1.28rem; font-weight:900; }}
.kpi-sub {{ color:{THEME["muted"]}; font-size:.84rem; }}
.compare-badge {{ display:inline-block; background:#FDE6F2; color:{THEME["magenta"]};
  font-weight:900; padding:2px 8px; border-radius:999px; margin-left:6px; }}

/* Segmented / Radio 가독성 */
[data-testid="stSegmentedControl"] [role="tab"] p {{ 
  color:{THEME["text"]}!important; font-weight:800!important; 
}}
[data-testid="stSegmentedControl"] [role="tab"][aria-selected="true"] p {{
  color:{THEME["primary"]}!important;
}}
[data-testid="stRadio"] label p {{ color:{THEME["text"]}!important; font-weight:800!important; }}
div[data-baseweb="button-group"] button {{
  background:#FFFFFF!important; color:{THEME["text"]}!important;
  border:1px solid {THEME["border"]}!important;
}}
div[data-baseweb="button-group"] button[aria-pressed="true"] {{
  background:#EEF2FF!important; color:{THEME["primary"]}!important; border:1px solid {THEME["primary"]}!important;
}}

/* Tabs */
div[role="tablist"] {{ gap: 14px; }}
div[role="tablist"] [role="tab"] p {{
  color: {THEME["text"]} !important; font-weight: 800 !important; opacity: 1 !important;
}}
div[role="tablist"] [role="tab"][aria-selected="true"] p {{ color: {THEME["primary"]} !important; }}
div[role="tablist"] [data-baseweb="tab-highlight"] {{
  background: {THEME["primary"]} !important; height: 3px !important; border-radius: 3px !important;
}}

/* ===== 주목 행정동 – 정사각 타일 그리드 ===== */
.topgrid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
  align-items: stretch;
}}
.top-tile {{
  background:{THEME["card"]}; border:1px solid {THEME["border"]};
  border-radius:16px; padding:16px; height:240px;
  box-shadow:0 8px 18px rgba(15,23,42,.06);
  display:flex; flex-direction:column;
}}
.top-tile .tit {{
  font-size:20px; font-weight:900; color:{THEME["accent"]};
  line-height:1.25; margin-bottom:8px; display:block;
  white-space:normal; word-break:keep-all; overflow-wrap:anywhere;
}}
.metric-list {{ list-style:none; padding:0; margin:4px 0 0; flex:1 1 auto; overflow:auto; }}
.metric-list li {{
  display:flex; align-items:center; gap:8px;
  margin:6px 0; padding:6px 10px; border-radius:10px;
  background:#F7FAFF; border:1px solid {THEME["border"]};
  color:{THEME["primary"]}; font-weight:800; font-size:13.5px;
}}
.metric-list .medal {{ font-size:14px; }}
.metric-list .rank  {{ color:{THEME["magenta"]}; font-weight:900; }}
.tile-badge {{
  align-self:flex-start; margin-top:8px;
  background:#FCE8F2; color:{THEME["magenta"]}; border:1px solid #FBD0E4;
  padding:4px 10px; border-radius:999px; font-weight:800; font-size:12px;
}}
.more-line {{ color:{THEME["muted"]}; font-weight:700; font-size:12.5px; }}

/* Streamlit 상단 헤더/툴바 숨기기 */
header[data-testid="stHeader"]{{display:none!important;}}
div[data-testid="stToolbar"]{{display:none!important;}}
#MainMenu, footer{{visibility:hidden;}}

/* 혹시 남아있는 st.json 디버그는 숨김 */
div[data-testid='stJson']{{display:none!important;}}
</style>
""", unsafe_allow_html=True)

st.title("✨ 부산시 청년 창업 유망 지역 인사이트")

# ============================= Mappings =============================
DISPLAY = {
    "visitors":"방문인구","residents":"주거인구","workers":"직장인구",
    "trx_amt":"총 소비금액","trx_cnt":"총 소비건수",
    "youth_amt":"청년 소비금액(20~30대)","youth_cnt":"청년 소비건수(20~30대)"
}
AGE_LABELS = {"under20":"20세미만","20s":"20대","30s":"30대","40s":"40대","50s":"50대","60s":"60대","over70":"70세이상"}
RANK_ICON = {1:"🥇", 2:"🥈", 3:"🥉"}

# ============================= Utils =============================
def fmt_kr(v: float, unit: str) -> str:
    v = float(v) if pd.notna(v) else 0.0
    if unit in ("원","건"):
        if abs(v) >= 1e8:  return f"{v/1e8:,.1f}억 {unit}"
        if unit=="건" and abs(v) >= 1e4: return f"{v/1e4:,.1f}만 {unit}"
    if unit=="명" and abs(v) >= 1e4: return f"{v/1e4:,.1f}만 명"
    return f"{int(round(v)):,} {unit}"

def apply_theme(fig, h=None):
    fig.update_layout(template="plotly_white",
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["card"],
        font=dict(color=THEME["text"]),
        xaxis=dict(gridcolor=THEME["grid"], tickfont=dict(color=THEME["muted"])),
        yaxis=dict(gridcolor=THEME["grid"], tickfont=dict(color=THEME["muted"])),
        legend=dict(bordercolor=THEME["border"], bgcolor=THEME["card"]),
        margin=dict(l=10,r=14,t=40,b=10), height=h)
    return fig

def safe_read_parquet(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(p)

def load_geojson(path: str):
    p = Path(path)
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"GeoJSON 로드 오류: {e}")
        return None

# ============================= ETL / Cache =============================
@st.cache_data
def parse_year_month(df: pd.DataFrame) -> pd.DataFrame:
    if {"연도","월"}.issubset(df.columns):
        df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
        df["월"] = pd.to_numeric(df["월"], errors="coerce").astype("Int64")
        return df
    if "기준년월" not in df.columns:
        raise ValueError("CSV에 '기준년월' 또는 '연도/월' 칼럼이 필요합니다.")
    base = df["기준년월"].astype(str).str.strip().str.replace(r"[./]", "-", regex=True)
    is_yyyymm = base.str.fullmatch(r"\d{6}", na=False)
    base = np.where(is_yyyymm, base.str[:4] + "-" + base.str[4:], base)
    dt = pd.to_datetime(pd.Series(base) + "-01", errors="coerce")
    df = df.loc[~dt.isna()].copy()
    df["연도"] = dt.dt.year.astype("Int64"); df["월"] = dt.dt.month.astype("Int64")
    return df

def detect_gender(col: str):
    s = col.lower()
    if re.search(r'(^|[_\s(])male\)?$', s) or '남성' in col or re.search(r'(_|)남$', col): return "male"
    if re.search(r'(^|[_\s(])female\)?$', s) or '여성' in col or re.search(r'(_|)여$', col): return "female"
    return None

def detect_metric(col: str):
    if "방문인구" in col or ("방문" in col and "인구" in col): return "visitors"
    if "주거인구" in col or ("주거" in col and "인구" in col): return "residents"
    if "직장인구" in col or ("직장" in col and "인구" in col): return "workers"
    if "금액" in col or "amt" in col.lower(): return "trx_amt"
    if "건수" in col or "cnt" in col.lower(): return "trx_cnt"
    return None

@st.cache_data
def build_cache(csv_path: str, cache_dir: str):
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = parse_year_month(df)

    # 시간대별: 방문/주거/직장/이용건수/이용금액
    rows=[]; bases_map={"방문인구":"visitors","주거인구":"residents","직장인구":"workers","이용건수":"trx_cnt","이용금액":"trx_amt"}
    for b, key in bases_map.items():
        cols = [c for c in df.columns if isinstance(c,str) and c.startswith(f"{b}_")]
        for c in cols:
            try: hour = int(str(c).split("_")[-1])
            except: continue
            sub = df[["행정동명","연도","월"]].copy()
            sub["지표"] = key; sub["시간"]=hour; sub["값"]=pd.to_numeric(df[c], errors="coerce")
            rows.append(sub)
    hourly = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["행정동명","연도","월","지표","시간","값"])
    hourly.to_parquet(cache/"hourly.parquet", index=False)

    # 연령대별(amt_*, cnt_*)
    a_rows=[]
    for c in [c for c in df.columns if isinstance(c,str) and (c.startswith("amt_") or c.startswith("cnt_"))]:
        sub = df[["행정동명","연도","월"]].copy()
        sub["지표"]=c; sub["값"]=pd.to_numeric(df[c], errors="coerce")
        a_rows.append(sub)
    age = pd.concat(a_rows, ignore_index=True) if a_rows else pd.DataFrame(columns=["행정동명","연도","월","지표","값"])
    age.to_parquet(cache/"age.parquet", index=False)

    # 성별
    g_rows=[]
    for c in df.columns:
        if not isinstance(c,str): continue
        g = detect_gender(c); m = detect_metric(c)
        if not g or not m: continue
        sub = df[["행정동명","연도","월"]].copy()
        sub["지표"]=m; sub["성별"]=g; sub["값"]=pd.to_numeric(df[c], errors="coerce")
        g_rows.append(sub)
    gender = pd.concat(g_rows, ignore_index=True) if g_rows else pd.DataFrame(columns=["행정동명","연도","월","지표","성별","값"])
    gender.to_parquet(cache/"gender.parquet", index=False)

def scope_filter(df: pd.DataFrame, year: int, month_sel: str):
    if df.empty: return df.copy(), f"{year}년"
    return (df[df["연도"]==year].copy() if month_sel=="(전체)" else df[(df["연도"]==year)&(df["월"]==month_sel)].copy(),
            f"{year}년 연평균" if month_sel=="(전체)" else f"{year}년 {month_sel}월")

def agg_city(scope_df: pd.DataFrame, metric_filter, month_sel: str):
    """도시 단위 집계
       - 시간대 원천(방문/주거/직장/이용금액/이용건수): 월별 합계
       - 연령대 파생(amt_*, cnt_*): 월별 합계
       - month=='(전체)'이면 행정동별 월값 평균(=연평균)을 사용
    """
    if scope_df.empty: return pd.DataFrame(columns=["행정동명","값"])

    if isinstance(metric_filter, list):
        sub = scope_df[scope_df["지표"].isin(metric_filter)].copy()
    elif isinstance(metric_filter, str) and metric_filter.endswith("*"):
        sub = scope_df[scope_df["지표"].str.startswith(metric_filter[:-1])].copy()
    else:
        sub = scope_df[scope_df["지표"]==metric_filter].copy()

    if sub.empty: return pd.DataFrame(columns=["행정동명","값"])

    # 월별 값 만들기
    if "시간" in sub.columns:
        # 시간대가 있으면 월합
        monthly = sub.groupby(["행정동명","월"], as_index=False)["값"].sum()
    else:
        # 연령대 파생(amt_/cnt_)은 본래 월 단위 → 여러 칼럼 합이면 월합
        monthly = sub.groupby(["행정동명","월"], as_index=False)["값"].sum()

    # 최종: 월 선택 vs 연-전체
    if month_sel == "(전체)":
        # 행정동별 월평균 (연평균)
        return monthly.groupby("행정동명", as_index=False)["값"].mean()
    else:
        # 선택월: 행정동별 해당월 합
        return monthly.groupby("행정동명", as_index=False)["값"].sum()

def city_agg_for_feature(metric_key: str, scope_h: pd.DataFrame, scope_a: pd.DataFrame, month_sel: str) -> pd.DataFrame:
    if   metric_key=="visitors": return agg_city(scope_h, "visitors", month_sel)
    elif metric_key in ("residents","workers","trx_cnt","trx_amt"):
        if not scope_h.empty and metric_key in scope_h.get("지표", pd.Series([])).unique():
            return agg_city(scope_h, metric_key, month_sel)  # 시간대 원천
        else:
            if metric_key=="trx_amt": return agg_city(scope_a, "amt_*", month_sel)
            if metric_key=="trx_cnt": return agg_city(scope_a, "cnt_*", month_sel)
            return pd.DataFrame(columns=["행정동명","값"])
    elif metric_key=="youth_amt": return agg_city(scope_a, ["amt_20s","amt_30s"], month_sel)
    elif metric_key=="youth_cnt": return agg_city(scope_a, ["cnt_20s","cnt_30s"], month_sel)
    return pd.DataFrame(columns=["행정동명","값"])

@st.cache_data
def get_top5_data(scope_h, scope_a, month_sel):
    res={}
    for k in ["visitors","residents","workers","trx_cnt","trx_amt","youth_amt","youth_cnt"]:
        d = city_agg_for_feature(k, scope_h, scope_a, month_sel)
        if not d.empty: res[k] = d.set_index("행정동명")["값"].nlargest(5)
    return res

def build_core_info(top5_dict):
    info = {}
    for mkey, series in top5_dict.items():
        if series.empty: continue
        for rank, (dong, _) in enumerate(series.items(), start=1):
            if dong not in info: info[dong] = {"count":0, "ranks":{}}
            info[dong]["count"] += 1
            info[dong]["ranks"][mkey] = rank
    return dict(sorted(info.items(), key=lambda x: (-x[1]["count"], x[0])))

# === 타일 렌더 도우미 ===
FEAT_LABEL = DISPLAY
FEAT_ORDER = ["visitors","residents","workers","trx_amt","trx_cnt","youth_amt","youth_cnt"]

def build_core_tiles_html(core_info: dict, min_count: int = 2, max_lines: int = 6) -> str:
    """앞 공백 없이 HTML 생성 → Markdown 코드블럭 방지. 타이틀 줄바꿈 허용."""
    items = [(dong,info) for dong,info in core_info.items() if info.get("count",0) >= min_count]
    items.sort(key=lambda kv: (-kv[1]["count"], min(kv[1]["ranks"].values()), kv[0]))
    if not items:
        return "<div></div>"

    tiles = []
    for dong, info in items:
        ordered = sorted(
            info["ranks"].items(),
            key=lambda kv: (kv[1], FEAT_ORDER.index(kv[0]) if kv[0] in FEAT_ORDER else 99)
        )
        lines = []
        for feat, rk in ordered[:max_lines]:
            medal = RANK_ICON.get(rk, "")
            label = FEAT_LABEL.get(feat, feat)
            lines.append(
                f"<li><span class='medal'>{medal}</span><span>{label}</span><span class='rank'>&nbsp;&nbsp;TOP {rk}</span></li>"
            )
        remain = max(0, len(ordered) - max_lines)
        more = f"<li class='more-line'>… 외 {remain}개</li>" if remain else ""

        tile = (
            "<div class='top-tile'>"
            f"<div class='tit' title='{dong}'>{dong}</div>"
            f"<ul class='metric-list'>{''.join(lines)}{more}</ul>"
            f"<div class='tile-badge'>{info['count']}개 지표 TOP5</div>"
            "</div>"
        )
        tiles.append(tile)

    return "<div class='topgrid'>" + "".join(tiles) + "</div>"

# ============================= Sidebar / Load =============================
with st.sidebar:
    st.header("⚙️ 데이터/범위")
    CSV  = st.text_input("CSV 경로", "통합_행정동_데이터_1st.csv")
    CACHE = st.text_input("캐시 경로", "./cache")
    GEO  = st.text_input("GeoJSON 경로", "busan_205.geojson")
    if st.button("🔄 캐시 생성/갱신"):
        with st.spinner("전처리 중..."): build_cache(CSV, CACHE)
        st.success("캐시 생성 완료")

    @st.cache_data(ttl=600, show_spinner="로딩...")
    def load_cache(path:str):
        p = Path(path)
        return (safe_read_parquet(p/"hourly.parquet"),
                safe_read_parquet(p/"age.parquet"),
                safe_read_parquet(p/"gender.parquet"))
    h,a,g = load_cache(CACHE)
    if h.empty and a.empty and g.empty:
        st.warning("캐시가 비어있습니다. CSV 확인 후 ‘캐시 생성/갱신’."); st.stop()

    years = sorted(pd.concat([d["연도"] for d in [h,a,g] if not d.empty]).dropna().unique(), reverse=True)
    year  = st.selectbox("연도", years)
    months = sorted(pd.concat([d.loc[d["연도"]==year,"월"] for d in [h,a,g] if not d.empty]).dropna().unique())
    month = st.selectbox("월", ["(전체)"] + list(months), index=0)
    gj = load_geojson(GEO)

# 상태 키 기본값
st.session_state.setdefault("dong_prefill", "(선택 안함)")

scope_h, scope_label = scope_filter(h, year, month)
scope_a, _ = scope_filter(a, year, month)
scope_g, _ = scope_filter(g, year, month)

# ============================= 0) TOP5 + Core =============================
st.markdown("<div class='h-section'>🏆 상위 행정동(지표별 TOP 5)</div>", unsafe_allow_html=True)
top5 = get_top5_data(scope_h, scope_a, month)

if top5:
    core_info = build_core_info(top5)

    # 주목 행정동 섹션
    st.markdown(
        "<div class='card'><div class='h-sub'>🌟 주목할 만한 행정동</div>"
        "<div style='color:#475569;'>선정 기준: <b>2개 이상</b> 지표에서 TOP5 (선택한 연/월 기준 자동 선별)</div></div>",
        unsafe_allow_html=True
    )

    if core_info:
        html_grid = build_core_tiles_html(core_info, min_count=2, max_lines=6)
        st.markdown(html_grid, unsafe_allow_html=True)
    else:
        st.info("여러 지표에서 동시에 두각을 나타내는 행정동이 없습니다.")

    # 지표별 TOP5 섹션 (구분)
    st.markdown("<div class='h-sub' style='margin-top:16px;'>📈 지표별 TOP5</div>", unsafe_allow_html=True)

    keys = list(DISPLAY.keys())
    for row in range(2):
        cols = st.columns(3, gap="large")
        for col, k in zip(cols, keys[row*3:(row+1)*3]):
            with col:
                title = f"{DISPLAY[k]}"
                if (k in top5) and (not top5[k].empty):
                    unit = "명" if k in ["visitors","residents","workers"] else ("원" if "amt" in k else "건")
                    badges = ["🥇","🥈","🥉","4.","5."]
                    items=[]
                    for i,(dong,val) in enumerate(top5[k].items()):
                        name_html = f"<b>{dong}</b>" if i<3 else dong
                        val_html  = f"<span class='top3-val'>{fmt_kr(val, unit)}</span>" if i<3 else fmt_kr(val, unit)
                        items.append(
                            f'<li><span class="rank-num">{badges[i]}</span>'
                            f'<span class="rank-name">{name_html}</span>'
                            f'<span class="rank-val">{val_html}</span></li>'
                        )
                    st.markdown(
                        f'<div class="card top5-card"><h5 class="h-sub">{title}</h5>'
                        f'<ol class="rank-list">{"".join(items)}</ol></div>',
                        unsafe_allow_html=True
                    )

st.divider()

# ============================= 1) 지도 + 요약 =============================
st.markdown("<div class='h-section'>🗺️ 지도 기반 탐색</div>", unsafe_allow_html=True)
left, right = st.columns([7,5], gap="large")

with left:
    metric_kor = st.selectbox("지도 지표", list(DISPLAY.values()), index=0, key="map_metric")
    metric = [k for k,v in DISPLAY.items() if v==metric_kor][0]
    dmap = city_agg_for_feature(metric, scope_h, scope_a, month)

    if gj and not dmap.empty:
        dmap["행정동명_지도용"] = dmap["행정동명"].astype(str).apply(lambda x: x.split()[-1])
        fig = px.choropleth_mapbox(
            dmap, geojson=gj, featureidkey="properties.ADM_NM",
            locations="행정동명_지도용", color="값", hover_name="행정동명",
            mapbox_style="carto-positron", center={"lat":35.18,"lon":129.07}, zoom=9.5,
            opacity=.88, color_continuous_scale=["#EBF2FF","#CFE1FF","#A7C4FF","#6FA0FF","#1E3A8A"],
            labels={'값': metric_kor})
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=540)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("GeoJSON 경로 또는 해당 지표 데이터가 필요합니다.")

with right:
    # 연-전체/월 안내
    scope_note = (
        f"<div class='kpi-sub' style='margin-bottom:8px;color:{THEME['muted']}'>"
        + ("선택한 <b>연평균</b> 기준 (행정동별 월값 평균 후 합산/평균 계산)" if month == "(전체)"
           else "선택한 <b>월</b> 기준 (해당 월의 모든 시간대/연령/성별 포함)")
        + "</div>"
    )
    st.markdown(f"<div class='h-sub'>📊 요약지표 & 그래프 — {metric_kor}</div>{scope_note}", unsafe_allow_html=True)

    if 'dmap' in locals() and not dmap.empty:
        total, mean = dmap["값"].sum(), dmap["값"].mean()
        topN = dmap.sort_values("값", ascending=False).head(10).reset_index(drop=True)
        top1_dong, top1_val = topN.loc[0,"행정동명"], topN.loc[0,"값"]
        top5_share = topN["값"].head(5).sum()/max(total,1)*100
        u = ("원" if "금액" in metric_kor else ("건" if "건수" in metric_kor else "명"))

        k1,k2 = st.columns(2, gap="small"); k3,k4 = st.columns(2, gap="small")
        for box, title, val, sub in [
            (k1,"전체 합계", fmt_kr(total, u), metric_kor),
            (k2,"상위 5개 집중도", f"{top5_share:,.1f}%", "도시 합계 대비"),
            (k3,"평균(행정동)", fmt_kr(mean, u), metric_kor),
            (k4,"최고 행정동", f"<b style='color:{THEME['magenta']}'>{top1_dong}</b>", f"<b style='color:{THEME['magenta']}'>{fmt_kr(top1_val, u)}</b>")
        ]:
            with box: st.markdown(f"<div class='card'><div class='kpi-title'>{title}</div><div class='kpi-value'>{val}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

        # 막대 (Top1 겹침 해결)
        colors = [THEME["pale"]]*len(topN); colors[0] = THEME["primary"]
        txt = topN["값"].apply(lambda v: f"{v:,.0f}"); txt.iloc[0] = ""
        fig1 = px.bar(topN, x="행정동명", y="값", text=txt, title=None)
        ylim = float(topN["값"].max()) * 1.35
        fig1.update_yaxes(range=[0, ylim])
        fig1.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
        fig1.update_traces(textposition="outside", marker_color=colors, cliponaxis=False)
        fig1.update_xaxes(tickangle=-25)
        fig1.add_annotation(
            x=top1_dong, y=top1_val*1.08,
            text=f"<b style='color:{THEME['magenta']}'>Top 1</b><br><b style='color:{THEME['magenta']}'>{fmt_kr(top1_val, u)}</b>",
            showarrow=False, align="center"
        )
        st.plotly_chart(apply_theme(fig1, h=360), use_container_width=True)

st.divider()

# ============================= 2) 행정동 상세 분석 =============================
st.markdown("<a id='detail'></a>", unsafe_allow_html=True)
st.markdown("<div class='h-section'>📎 행정동 상세 분석</div>", unsafe_allow_html=True)

all_dongs = sorted(pd.concat([d["행정동명"] for d in [h,a,g] if not d.empty]).dropna().unique())
prefill = st.session_state.get("dong_prefill","(선택 안함)")
options = ["(선택 안함)"] + list(all_dongs)
default_index = options.index(prefill) if prefill in options else 0

c1,c2 = st.columns([2,3], gap="large")
with c1:
    DONG_SELECT_KEY = "dong_sel_main"
    dong = st.selectbox("분석할 행정동", options, index=default_index, key=DONG_SELECT_KEY)
with c2:
    try:
        feat_kor = st.segmented_control("피쳐 선택", list(DISPLAY.values()), selection=list(DISPLAY.values())[0], key="feat_picker_main")
    except Exception:
        feat_kor = st.radio("피쳐 선택", list(DISPLAY.values()), index=0, horizontal=True, key="feat_picker_main")
feat = [k for k,v in DISPLAY.items() if v==feat_kor][0]

if dong != "(선택 안함)":
    city_agg = city_agg_for_feature(feat, scope_h, scope_a, month)

    if not city_agg.empty and (dong in city_agg["행정동명"].values):
        n = int(city_agg["행정동명"].nunique())
        city_agg_sorted = (
            city_agg.sort_values("값", ascending=False)
                    .drop_duplicates("행정동명")
                    .reset_index(drop=True)
        )
        try:
            rank = int(city_agg_sorted.index[city_agg_sorted["행정동명"] == dong].item()) + 1
        except ValueError:
            rank = n
        val = float(city_agg_sorted.loc[rank - 1, "값"])

        vals = city_agg_sorted["값"].astype(float).to_numpy()
        num_lt = int((vals < val).sum())
        num_eq = int((vals == val).sum())
        pct_bottom = (num_lt + 0.5 * num_eq) / n * 100.0
        pct_top = float(np.clip(100.0 - pct_bottom, 0.0, 100.0))

        unit = "원" if "금액" in feat_kor else ("건" if "건수" in feat_kor else "명")

        cA, cB, cC = st.columns(3, gap="large")
        with cA:
            st.markdown(
                f"<div class='card'>"
                f"<div class='kpi-title'>도시 내 순위</div>"
                f"<div class='kpi-value'>{rank} / {n}<span class='compare-badge'>상위 {pct_top:.1f}%</span></div></div>",
                unsafe_allow_html=True
            )
        with cB:
            st.markdown(
                f"<div class='card'><div class='kpi-title'>해당 값</div>"
                f"<div class='kpi-value' style='color:{THEME['magenta']}'>{fmt_kr(val, unit)}</div></div>",
                unsafe_allow_html=True
            )
        with cC:
            diff = val - city_agg['값'].mean()
            sign = '▲' if diff >= 0 else '▼'
            st.markdown(
                f"<div class='card'><div class='kpi-title'>평균 대비</div>"
                f"<div class='kpi-value'>{sign} {fmt_kr(abs(diff), unit)}</div></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("선택한 행정동의 집계 데이터가 없습니다.")

    tabs = st.tabs(["시간대","연령대","성별","월별 추이"])

    # --- 시간대
    with tabs[0]:
        if not scope_h.empty and (feat in scope_h.get("지표", pd.Series([])).unique()):
            d = scope_h[(scope_h["행정동명"]==dong) & (scope_h["지표"]==feat)]
            m = d.groupby("시간", as_index=False)["값"].mean()
            if not m.empty:
                y_max_idx = m["값"].idxmax()
                fig = px.line(m, x="시간", y="값", markers=True, title=None)
                fig.update_traces(line=dict(color=THEME["primary"]), marker=dict(color=THEME["primary"]))
                fig.add_scatter(x=[m.loc[y_max_idx,"시간"]], y=[m.loc[y_max_idx,"값"]],
                                mode="markers+text", text=[f"<b style='color:{THEME['magenta']}'>Peak</b>"],
                                textposition="top center", marker=dict(size=11, color=THEME["primary"]))
                st.plotly_chart(apply_theme(fig, 320), use_container_width=True)
            else:
                st.caption("이 피처는 시간대 데이터가 없습니다.")
        else:
            st.caption("이 피처는 시간대 데이터가 없습니다.")

    # --- 연령대
    with tabs[1]:
        if feat in ["trx_amt","trx_cnt","youth_amt","youth_cnt"] and not scope_a.empty:
            if "youth" in feat:
                cols = ["amt_20s","amt_30s"] if feat=="youth_amt" else ["cnt_20s","cnt_30s"]
                d = scope_a[(scope_a["행정동명"]==dong) & (scope_a["지표"].isin(cols))]
                m = d.groupby("지표", as_index=False)["값"].mean()
                m["연령대"] = m["지표"].map({"amt_20s":"20대","amt_30s":"30대","cnt_20s":"20대","cnt_30s":"30대"})
            else:
                pref = "amt_" if feat=="trx_amt" else "cnt_"
                d = scope_a[(scope_a["행정동명"]==dong) & (scope_a["지표"].str.startswith(pref))]
                m = d.groupby("지표", as_index=False)["값"].mean()
                m["연령대"] = m["지표"].str.replace(r"^(amt|cnt)_","", regex=True).map(AGE_LABELS)
            if not m.empty:
                idx = int(m["값"].idxmax())
                colors = [THEME["pale"]]*len(m); colors[list(m.index).index(idx)] = THEME["primary"]
                fig = px.bar(m, x="연령대", y="값", text="값", title=None)
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', marker_color=colors, cliponaxis=False)
                st.plotly_chart(apply_theme(fig, 340), use_container_width=True)
            else:
                st.caption("이 피처는 연령대 그래프가 없습니다.")
        else:
            st.caption("이 피처는 연령대 그래프가 없습니다.")

    # --- 성별
    with tabs[2]:
        if feat in ["visitors","residents","workers","trx_amt","trx_cnt"] and not scope_g.empty:
            d = scope_g[(scope_g["행정동명"]==dong) & (scope_g["지표"]==feat)]
            if not d.empty:
                m = d.groupby("성별", as_index=False)["값"].mean()
                m["성별"] = m["성별"].map({"male":"남성","female":"여성"})
                fig = px.pie(m, names="성별", values="값", title=None,
                             color="성별", color_discrete_map={"남성": THEME["pale"], "여성": THEME["magenta"]})
                fig.update_traces(textinfo="percent+label", hole=.35)
                st.plotly_chart(apply_theme(fig, 320), use_container_width=True)
            else:
                st.caption("성별 데이터가 없습니다.")
        else:
            st.caption("이 피처는 성별 그래프가 없습니다.")

    # --- 월별 추이
    with tabs[3]:
        if feat in scope_h.get("지표", pd.Series([])).unique():
            d = scope_h[(scope_h["행정동명"]==dong) & (scope_h["지표"]==feat)]
            m = d.groupby("월", as_index=False)["값"].mean()
        elif feat in ["trx_amt","trx_cnt"]:
            pref = "amt_" if feat=="trx_amt" else "cnt_"
            d = scope_a[(scope_a["행정동명"]==dong) & (scope_a["지표"].str.startswith(pref))]
            m = d.groupby("월", as_index=False)["값"].sum()
        else:
            cols = ["amt_20s","amt_30s"] if feat=="youth_amt" else ["cnt_20s","cnt_30s"]
            d = scope_a[(scope_a["행정동명"]==dong) & (scope_a["지표"].isin(cols))]
            m = d.groupby("월", as_index=False)["값"].sum()
        if not m.empty:
            y_max_idx = m["값"].idxmax()
            fig = px.line(m.sort_values("월"), x="월", y="값", markers=True, title=None)
            fig.update_traces(line=dict(color=THEME["primary"]), marker=dict(color=THEME["primary"]))
            fig.add_scatter(x=[m.loc[y_max_idx,"월"]], y=[m.loc[y_max_idx,"값"]],
                            mode="markers+text", text=[f"<b style='color:{THEME['magenta']}'>Peak</b>"],
                            textposition="top center", marker=dict(size=11, color=THEME["primary"]))
            st.plotly_chart(apply_theme(fig, 330), use_container_width=True)
        else:
            st.caption("월별 추이 데이터가 없습니다.")
