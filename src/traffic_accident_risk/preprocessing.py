"""Preprocessing and feature engineering pipeline extracted from LG.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess_dataframe(
    df: pd.DataFrame,
    drop_list: list[str],
    is_train: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run preprocessing + engineered features."""
    frame = df.copy()

    def remove_invalid_rows(local_df: pd.DataFrame, train_mode: bool = True) -> pd.DataFrame:
        if not train_mode:
            return local_df

        condition_reasons = [
            "기증용",
            "기증용, 난자 저장용",
            "기증용, 배아 저장용",
            "난자 저장용",
            "난자 저장용, 배아 저장용",
            "난자 저장용, 배아 저장용, 연구용",
            "배아 저장용",
        ]

        delete_mask = (
            local_df["배아 생성 주요 이유"].isin(condition_reasons)
            & (local_df["이식된 배아 수"] == 0)
            & (local_df["임신 성공 여부"] == 1)
        )
        return local_df[~delete_mask].reset_index(drop=True)

    def preprocess_missing_values(local_df: pd.DataFrame) -> pd.DataFrame:
        di_fill_columns = [
            "단일 배아 이식 여부",
            "착상 전 유전 진단 사용 여부",
            "총 생성 배아 수",
            "미세주입된 난자 수",
            "미세주입에서 생성된 배아 수",
            "이식된 배아 수",
            "미세주입 배아 이식 수",
            "미세주입 후 저장된 배아 수",
            "해동된 배아 수",
            "해동 난자 수",
            "저장된 신선 난자 수",
            "수집된 신선 난자 수",
            "저장된 배아 수",
            "혼합된 난자 수",
            "파트너 정자와 혼합된 난자 수",
            "기증자 정자와 혼합된 난자 수",
            "동결 배아 사용 여부",
            "신선 배아 사용 여부",
            "기증 배아 사용 여부",
            "대리모 여부",
        ]

        if "시술 유형" in local_df.columns:
            di_mask = local_df["시술 유형"] == "DI"
            existing = [col for col in di_fill_columns if col in local_df.columns]
            local_df.loc[di_mask, existing] = local_df.loc[di_mask, existing].fillna(0)

        categorical_cols = local_df.select_dtypes(include=["object"]).columns
        local_df[categorical_cols] = local_df[categorical_cols].where(
            local_df[categorical_cols].notna(), "Unknown"
        )
        if "PGD 시술 여부" in local_df.columns:
            local_df["PGD 시술 여부"] = local_df["PGD 시술 여부"].fillna(0)

        required = {"배아 이식 경과일", "동결 배아 사용 여부", "기증 배아 사용 여부", "난자 혼합 경과일"}
        if required.issubset(set(local_df.columns)):
            condition = (
                (local_df["배아 이식 경과일"].between(0, 6))
                & (local_df["동결 배아 사용 여부"] == 0)
                & (local_df["기증 배아 사용 여부"] == 0)
            )
            local_df.loc[condition, "난자 혼합 경과일"] = local_df.loc[
                condition, "난자 혼합 경과일"
            ].fillna(0)

        return local_df

    def generate_features(local_df: pd.DataFrame) -> pd.DataFrame:
        def convert_to_numeric(value: object) -> int:
            if value == "6회 이상":
                return 6
            try:
                return int(str(value).replace("회", ""))
            except Exception:
                return 0

        def safe_divide(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
            return np.where(denominator == 0, 0, numerator / denominator)

        numeric_cols = [
            "DI 시술 횟수",
            "DI 임신 횟수",
            "DI 출산 횟수",
            "IVF 시술 횟수",
            "IVF 임신 횟수",
            "IVF 출산 횟수",
            "총 시술 횟수",
            "클리닉 내 총 시술 횟수",
        ]
        for col in numeric_cols:
            local_df[col] = local_df[col].astype(str).apply(convert_to_numeric)

        local_df["총 시술 횟수"] = local_df["IVF 시술 횟수"] + local_df["DI 시술 횟수"]
        local_df["총 임신 횟수"] = local_df["IVF 임신 횟수"] + local_df["DI 임신 횟수"]
        local_df["총 출산 횟수"] = local_df["IVF 출산 횟수"] + local_df["DI 출산 횟수"]

        local_df["IVF 임신 대비 출산율"] = (
            local_df["IVF 출산 횟수"] / local_df["IVF 임신 횟수"]
        ).fillna(0)
        local_df["DI 임신 대비 출산율"] = (
            local_df["DI 출산 횟수"] / local_df["DI 임신 횟수"]
        ).fillna(0)

        local_df["출산 성공률"] = safe_divide(local_df["총 출산 횟수"], local_df["총 임신 횟수"])
        local_df["저장된 배아 수 대비 이식된 배아 비율"] = safe_divide(
            local_df["이식된 배아 수"], local_df["저장된 배아 수"]
        )
        local_df["해동 배아 비율"] = safe_divide(local_df["해동된 배아 수"], local_df["저장된 배아 수"])
        local_df["미세주입 비율"] = safe_divide(
            local_df["미세주입된 난자 수"], local_df["수집된 신선 난자 수"]
        )
        local_df["혼합된 난자 비율"] = safe_divide(
            local_df["혼합된 난자 수"], local_df["수집된 신선 난자 수"]
        )

        infertility_columns = [
            "불임 원인 - 난관 질환",
            "불임 원인 - 남성 요인",
            "불임 원인 - 배란 장애",
            "불임 원인 - 여성 요인",
            "불임 원인 - 자궁경부 문제",
            "불임 원인 - 자궁내막증",
            "남성 주 불임 원인",
            "남성 부 불임 원인",
            "여성 주 불임 원인",
            "여성 부 불임 원인",
            "부부 주 불임 원인",
            "부부 부 불임 원인",
        ]
        existing_columns = [col for col in infertility_columns if col in local_df.columns]
        local_df["불임 원인 총합"] = local_df[existing_columns].sum(axis=1)

        local_df["배아 생성 대비 저장된 배아 비율"] = safe_divide(
            local_df["저장된 배아 수"], local_df["총 생성 배아 수"]
        )
        local_df["배아 생성 대비 해동된 배아 비율"] = safe_divide(
            local_df["해동된 배아 수"], local_df["총 생성 배아 수"]
        )
        local_df["버려진 배아 비율"] = safe_divide(
            local_df["총 생성 배아 수"] - local_df["이식된 배아 수"] - local_df["저장된 배아 수"],
            local_df["총 생성 배아 수"],
        )
        local_df["IVF 사용 빈도"] = safe_divide(local_df["IVF 시술 횟수"], local_df["총 시술 횟수"])
        local_df["DI 사용 빈도"] = safe_divide(local_df["DI 시술 횟수"], local_df["총 시술 횟수"])

        local_df["남성 불임 원인 총합"] = local_df[
            ["남성 주 불임 원인", "남성 부 불임 원인", "불임 원인 - 남성 요인"]
        ].sum(axis=1)
        local_df["여성 불임 원인 총합"] = local_df[
            [
                "여성 주 불임 원인",
                "여성 부 불임 원인",
                "불임 원인 - 여성 요인",
                "불임 원인 - 배란 장애",
                "불임 원인 - 자궁경부 문제",
                "불임 원인 - 자궁내막증",
            ]
        ].sum(axis=1)
        local_df["부부 불임 원인 총합"] = local_df[
            ["부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인"]
        ].sum(axis=1)
        local_df["정자 문제 종합"] = local_df[
            [
                "불임 원인 - 정자 농도",
                "불임 원인 - 정자 면역학적 요인",
                "불임 원인 - 정자 운동성",
                "불임 원인 - 정자 형태",
            ]
        ].sum(axis=1)

        def categorize_uterus_condition(row: pd.Series) -> str:
            if row["총 생성 배아 수"] != 0 and row["저장된 배아 수"] != 0:
                return "좋음" if row["이식된 배아 수"] != 0 else "좋지 않음"
            return "알 수 없음"

        local_df["환자의 현재 자궁 상태"] = local_df.apply(categorize_uterus_condition, axis=1)
        local_df["자연 인공 혼합 생성 배아 수"] = (
            local_df["총 생성 배아 수"] - local_df["미세주입에서 생성된 배아 수"] - local_df["저장된 배아 수"]
        )
        local_df["저장된 자연 인공 혼합 배아 수"] = (
            local_df["저장된 배아 수"] - local_df["미세주입 후 저장된 배아 수"]
        )

        fresh_mask = (local_df["동결 배아 사용 여부"] == 0) & (local_df["기증 배아 사용 여부"] == 0)
        local_df.loc[fresh_mask, "자연 인공 혼합 배아 이식 수"] = (
            local_df["이식된 배아 수"] - local_df["미세주입 배아 이식 수"]
        )
        local_df.loc[fresh_mask, "생성한 배아 이식 수"] = (
            local_df["미세주입 배아 이식 수"] + local_df["자연 인공 혼합 배아 이식 수"]
        )
        local_df.loc[fresh_mask, "생성된 품질 좋은 배아 수"] = (
            local_df["저장된 배아 수"] + local_df["생성한 배아 이식 수"]
        )
        local_df.loc[
            ~fresh_mask,
            ["자연 인공 혼합 배아 이식 수", "생성한 배아 이식 수", "생성된 품질 좋은 배아 수"],
        ] = np.nan

        local_df["유산 횟수"] = local_df["총 임신 횟수"] - local_df["총 출산 횟수"]

        def categorize_stored_embryo_count(n: float) -> str:
            if 0 < n <= 6:
                return "최적 범위"
            if 6 < n <= 12:
                return "저위험"
            if n > 12:
                return "고위험"
            return "알 수 없음"

        local_df["저장된 배아 수(위험도)"] = local_df["저장된 배아 수"].apply(
            categorize_stored_embryo_count
        )

        def categorize_mixed_egg_count(n: float) -> str:
            if 0 < n <= 3 or 30 <= n <= 40:
                return "고위험"
            if 4 <= n <= 7 or 25 <= n <= 29:
                return "저위험"
            if 8 <= n <= 24:
                return "적정"
            return "알 수 없음"

        local_df["혼합된 난자 수(위험도)"] = local_df["혼합된 난자 수"].apply(categorize_mixed_egg_count)

        def categorize_culture_days(days: float) -> str:
            if days < 0:
                return "알 수 없음"
            if days in [1, 2, 3]:
                return "3일 배양"
            if days in [4, 5, 6, 7]:
                return "5일 배양"
            return "알 수 없음"

        local_df["신선 배아 배양 일 수"] = local_df["배아 이식 경과일"] - local_df["난자 혼합 경과일"]
        local_df["신선 배아 배양 유형"] = local_df["신선 배아 배양 일 수"].apply(categorize_culture_days)
        local_df["동결 배아 배양 일 수"] = local_df["배아 이식 경과일"] - local_df["난자 해동 경과일"]
        local_df["동결 배아 배양 유형"] = local_df["동결 배아 배양 일 수"].apply(categorize_culture_days)
        local_df["배아 배양 유형"] = np.where(
            local_df["동결 배아 배양 유형"] != "알 수 없음",
            local_df["동결 배아 배양 유형"],
            local_df["신선 배아 배양 유형"],
        )

        local_df["현재 시술용 여부"] = local_df["배아 생성 주요 이유"].apply(
            lambda value: "1" if isinstance(value, str) and "현재 시술용" in value.split(",") else "0"
        )

        def categorize_embryo_implantation_rate(rate: float) -> str:
            if rate > 100:
                return "알 수 없음"
            if 30 <= rate <= 100:
                return "1"
            if 0 < rate < 30:
                return "0"
            return "알 수 없음"

        local_df["배아 이식 대비 생성 비율"] = (
            (local_df["이식된 배아 수"] / local_df["총 생성 배아 수"])
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            * 100
        ).astype(int)
        local_df["배아 이식 대비 생성 비율 분류"] = local_df["배아 이식 대비 생성 비율"].apply(
            categorize_embryo_implantation_rate
        )
        local_df = local_df.drop(columns="배아 이식 대비 생성 비율")

        def categorize_embryo_risk(total_embryos: float) -> str:
            return "고위험" if total_embryos < 5 else "중간 위험"

        local_df["배아 품질 위험 지수(생성 배아 수 기준)"] = local_df["총 생성 배아 수"].apply(
            categorize_embryo_risk
        )

        def categorize_egg_count(n: float) -> str:
            if n < 10:
                return "고위험 (10개 미만)"
            if 10 <= n <= 15:
                return "최적 범위 (10~15개)"
            if 16 <= n <= 20:
                return "저위험 (16~20개)"
            return "중위험 (20개 초과)"

        local_df["난자 채취 범주"] = local_df["수집된 신선 난자 수"].apply(categorize_egg_count)

        local_df["PGS_검사진단"] = (
            local_df[["착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부", "PGS 시술 여부"]]
            .fillna(0)
            .sum(axis=1)
        )
        local_df["PGD_검사진단"] = (
            local_df[["착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부", "PGD 시술 여부"]]
            .fillna(0)
            .sum(axis=1)
        )

        local_df.loc[
            local_df["특정 시술 유형"].str.contains("BLASTOCYST", case=False, na=False),
            "배아 배양 유형",
        ] = "5일 배양"

        local_df["is_5day_culture"] = (local_df["배아 배양 유형"] == "5일 배양").astype(int)
        local_df["is_pgs"] = (local_df["PGS_검사진단"] > 0).astype(int)
        local_df["is_pgd"] = (local_df["PGD_검사진단"] > 0).astype(int)
        local_df["is_under_35"] = (local_df["시술 당시 나이"] == "만18-34세").astype(int)
        local_df["combined_feature"] = (
            local_df["is_5day_culture"]
            + local_df["is_pgs"]
            + local_df["is_pgd"]
            + local_df["단일 배아 이식 여부"]
            + local_df["is_under_35"]
        )

        unknown_icsi_mask = (
            (local_df["특정 시술 유형"] == "Unknown")
            & (
                (local_df["미세주입된 난자 수"] != 0)
                | (local_df["미세주입에서 생성된 배아 수"] != 0)
                | (local_df["미세주입 배아 이식 수"] != 0)
                | (local_df["미세주입 후 저장된 배아 수"] != 0)
            )
        )
        local_df.loc[unknown_icsi_mask, "특정 시술 유형"] = "ICSI"

        local_df["불임원인-정자"] = (
            local_df["불임 원인 - 정자 농도"]
            + local_df["불임 원인 - 정자 면역학적 요인"]
            + local_df["불임 원인 - 정자 운동성"]
            + local_df["불임 원인 - 정자 형태"]
        )

        if verbose:
            print("\n📌 [파생 변수 생성 완료]\n")
        return local_df

    def print_current_features(local_df: pd.DataFrame) -> None:
        if not verbose:
            return
        print("\n🔹 현재 학습에 사용되는 변수 목록:")
        print(f"총 {local_df.shape[1]}개 변수 사용 중")
        print(", ".join(local_df.columns))

    frame = preprocess_missing_values(frame)
    frame = remove_invalid_rows(frame, train_mode=is_train)
    frame = generate_features(frame)
    frame.drop(columns=drop_list, errors="ignore", inplace=True)
    print_current_features(frame)

    cols_to_convert = [col for col in frame.columns if col != "임신 성공 여부"]
    frame = frame.astype({col: str for col in cols_to_convert})
    return frame

