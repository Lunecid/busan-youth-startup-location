# -*- coding: utf-8 -*-
"""
Busan 행정동 통합 데이터 전처리 유틸
- 연도/월 파생
- 성별 지표 long 변환
- 시간대별 지표 long 변환(방문/주거/직장/이용건수/이용금액)
- 연령대별(amt/cnt) long 변환
- 캐시 저장/로딩
사용 예:
python dash_preprocess.py --csv 통합_행정동_데이터_1st.csv --out ./cache
"""
import pandas as pd
import re
from pathlib import Path

KEYS = ["기준년월", "행정동코드", "행정동명"]

GENDER_BASES = [
    ("평균주거인구수", "residents"),
    ("평균직장인구수", "workers"),
    ("평균방문인구수", "visitors"),
    ("평균이용건수",   "trx_cnt"),
    ("평균이용금액",   "trx_amt"),
]

HOUR_BASES = [
    ("방문인구_",  "visitors"),
    ("주거인구_",  "residents"),
    ("직장인구_",  "workers"),
    ("이용건수_",  "trx_cnt"),
    ("이용금액_",  "trx_amt"),
]

AGE_AMT_RE = re.compile(r"^amt_(under20|[2-6]0s|over70)$")
AGE_CNT_RE = re.compile(r"^cnt_(under20|[2-6]0s|over70)$")

def add_year_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["연도"] = out["기준년월"].str.slice(0, 4).astype(int)
    out["월"]   = out["기준년월"].str.slice(5, 7).astype(int)
    return out

def melt_gender(df: pd.DataFrame) -> pd.DataFrame:
    out = add_year_month(df)
    rows = []
    for kor_name, std_name in GENDER_BASES:
        m = f"{kor_name}_male"
        f = f"{kor_name}_female"
        if m in out.columns and f in out.columns:
            tmp = out[KEYS + ["연도","월", m, f]].copy()
            tmp = tmp.melt(
                id_vars=KEYS + ["연도","월"],
                value_vars=[m, f],
                var_name="성별",
                value_name="값"
            )
            tmp["성별"] = tmp["성별"].map({m: "male", f: "female"})
            tmp["지표"] = std_name
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=KEYS + ["연도","월","성별","지표","값"])
    return pd.concat(rows, ignore_index=True)

def melt_hourly(df: pd.DataFrame) -> pd.DataFrame:
    out = add_year_month(df)
    frames = []
    for prefix, std_name in HOUR_BASES:
        hour_cols = [c for c in out.columns if c.startswith(prefix)]
        if not hour_cols:
            continue
        tmp = out[KEYS + ["연도","월"] + hour_cols].copy()
        m = tmp.melt(id_vars=KEYS + ["연도","월"], var_name="원컬럼", value_name="값")
        # 끝 2자리에서 시간 추출 (..00~23)
        m["시간"] = m["원컬럼"].str.extract(r"(\d{2})$").astype(int)
        m["지표"] = std_name
        frames.append(m.drop(columns=["원컬럼"]))
    if not frames:
        return pd.DataFrame(columns=KEYS + ["연도","월","시간","지표","값"])
    return pd.concat(frames, ignore_index=True)

def melt_age(df: pd.DataFrame) -> pd.DataFrame:
    out = add_year_month(df)
    amt_cols = [c for c in out.columns if AGE_AMT_RE.match(c)]
    cnt_cols = [c for c in out.columns if AGE_CNT_RE.match(c)]
    rows = []
    if amt_cols:
        t = out[KEYS + ["연도","월"] + amt_cols].copy()
        a = t.melt(id_vars=KEYS + ["연도","월"], var_name="연령대", value_name="값")
        a["유형"] = "amt"; rows.append(a)
    if cnt_cols:
        t = out[KEYS + ["연도","월"] + cnt_cols].copy()
        c = t.melt(id_vars=KEYS + ["연도","월"], var_name="연령대", value_name="값")
        c["유형"] = "cnt"; rows.append(c)
    if not rows:
        return pd.DataFrame(columns=KEYS + ["연도","월","연령대","유형","값"])
    ag = pd.concat(rows, ignore_index=True)
    ag["연령대"] = ag["연령대"].str.replace("amt_","",regex=False).str.replace("cnt_","",regex=False)
    return ag

def build_cache(csv_path: str, cache_dir: str = "./cache") -> dict:
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(csv_path, encoding="utf-8")
    g = melt_gender(raw)
    h = melt_hourly(raw)
    a = melt_age(raw)

    g.to_parquet(Path(cache_dir) / "gender.parquet", index=False)
    h.to_parquet(Path(cache_dir) / "hourly.parquet", index=False)
    a.to_parquet(Path(cache_dir) / "age.parquet", index=False)

    meta = {
        "n_rows": len(raw),
        "n_cols": len(raw.columns),
        "years": sorted(raw["기준년월"].str.slice(0,4).astype(int).unique().tolist()),
        "n_dong": raw["행정동명"].nunique(),
        "cache_dir": str(Path(cache_dir).resolve())
    }
    Path(cache_dir, "meta.json").write_text(
        pd.Series(meta).to_json(force_ascii=False, indent=2),
        encoding="utf-8"
    )
    return meta

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to CSV")
    p.add_argument("--out", default="./cache", help="cache dir")
    args = p.parse_args()
    info = build_cache(args.csv, args.out)
    print(info)
