import pandas as pd
import json
from pathlib import Path

# --- 설정: 이 두 줄의 파일 경로를 자신의 환경에 맞게 정확히 수정해주세요 ---
CSV_PATH = "통합_행정동_데이터_1st.csv"
GEOJSON_PATH = "busan_205.geojson"


# --------------------------------------------------------------------

def run_test():
    # 1. GeoJSON 파일 로드 및 이름 추출
    try:
        gj = json.loads(Path(GEOJSON_PATH).read_text(encoding="utf-8"))
        geojson_names = {f['properties']['ADM_NM'] for f in gj['features']}
        print(f"✅ GeoJSON 로드 성공: '{GEOJSON_PATH}' (총 {len(geojson_names)}개 지역)")
    except Exception as e:
        print(f"❌ GeoJSON 파일 로드 실패: '{GEOJSON_PATH}'")
        print(f"   오류: {e}")
        return

    # 2. CSV 파일 로드 및 이름 추출/가공
    try:
        # dash_preprocess.py가 parquet 파일을 생성하므로, parquet을 직접 읽는 것이 더 빠릅니다.
        # hourly.parquet 파일이 cache 폴더 안에 있는지 확인해주세요.
        df = pd.read_parquet("./cache/hourly.parquet")

        # '구/군' 제거하여 지도와 매칭할 이름 생성
        csv_names_processed = set(df['행정동명'].apply(lambda x: x.split()[-1]))

        print(f"✅ CSV 데이터 로드 성공 (./cache/hourly.parquet)")
        print(f"   - 원본 이름 샘플: {set(df['행정동명'].head(3))}")
        print(f"   - 가공된 이름 샘플: {[name for name in list(csv_names_processed)[:3]]}")
    except Exception as e:
        print(f"❌ CSV 데이터(Parquet) 로드 실패: './cache/hourly.parquet'")
        print(f"   오류: {e}")
        print(f"   '캐시 생성/갱신' 버튼을 눌러 cache 폴더가 생성되었는지 확인하세요.")
        return

    # 3. 두 데이터의 이름 비교
    matched_names = geojson_names.intersection(csv_names_processed)

    print("\n--- [ 최종 매핑 테스트 결과 ] ---")
    if not matched_names:
        print("❌ 매칭 성공 0개. 이름이 하나도 일치하지 않습니다.")

        unmatched_geojson = list(geojson_names - csv_names_processed)[:5]
        unmatched_csv = list(csv_names_processed - geojson_names)[:5]

        print("\n[비교 샘플]")
        print(f"  - GeoJSON 이름: {unmatched_geojson}")
        print(f"  - CSV 가공 후 이름: {unmatched_csv}")
    else:
        print(f"🎉 총 {len(matched_names)} / {len(geojson_names)} 개의 이름 매칭 성공!")
        print(f"   매칭된 이름 샘플: {list(matched_names)[:5]}")


if __name__ == "__main__":
    run_test()