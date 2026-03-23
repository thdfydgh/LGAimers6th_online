# 라이브러리
########################################################
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score)
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
#########################################################

# 전처리 + 파생 변수 생성 함수

def Preprocessing(df, drop_list, is_train = False):
    '''
    1. 조건부 데이터 일부 삭제 (별로 없음 10개 이하였음)
    이유: 현재 시술용이 아니면서 이식된 배아가 없는데 임신 성공 여부가 1로 존재한 행
    도메인적으로 이해가 안됨. -> 필요없는 행이라고 생각함.
    '''
    # 1. 필요 없는 행 지우는 함수 (train에만 적용)  - 전처리
    def remove_invalid_rows(df, is_train = True):

        if not is_train:
            return df

        조건_배아_생성_이유 = [
            '기증용', '기증용, 난자 저장용', '기증용, 배아 저장용',
            '난자 저장용', '난자 저장용, 배아 저장용',
            '난자 저장용, 배아 저장용, 연구용', '배아 저장용'
            ]
        
        삭제_조건 = (df['배아 생성 주요 이유'].isin(조건_배아_생성_이유)) & \
                  (df['이식된 배아 수'] == 0) & (df['임신 성공 여부'] == 1)

        return df[~삭제_조건].reset_index(drop=True)
        
    ######################################################################################3
    
    # 2. 결측치 처리 함수 - 전처리
    
    def preprocess_missing_values(df):
        '''
        1. DI 시술 관련 결측치 0으로 처리
        2. object 타입 결측치 unknown으로 처리
        3. PGD 시술 여부 값 결측치 0으로 처리
        4. 난자 혼합 경과일 조건부 설정으로 일부 결측치 0으로 처리
        '''
        
        
        
    
        # 2. '시술 유형'이 'DI'일 때 특정 컬럼(cols)의 결측치를 0으로 처리
        cols = [
            '단일 배아 이식 여부', '착상 전 유전 진단 사용 여부', 
            '총 생성 배아 수', '미세주입된 난자 수', '미세주입에서 생성된 배아 수',
            '이식된 배아 수', '미세주입 배아 이식 수', '미세주입 후 저장된 배아 수',
            '해동된 배아 수', '해동 난자 수', '저장된 신선 난자 수', '수집된 신선 난자 수' ,'저장된 배아 수', '혼합된 난자 수',
         '파트너 정자와 혼합된 난자 수', '기증자 정자와 혼합된 난자 수',
            '동결 배아 사용 여부', '신선 배아 사용 여부', '기증 배아 사용 여부', '대리모 여부'
        ] # '수집된 신선 난자 수' ,'저장된 배아 수', '혼합된 난자 수',
    
        if '시술 유형' in df.columns:
            di_mask = df['시술 유형'] == 'DI'  # '시술 유형'이 'DI'인 행 필터링
            df.loc[di_mask, cols] = df.loc[di_mask, cols].fillna(0)
            
    
        # 3. object type (임신 시도 경과일, 특정 시술 유형 ) -> 2개의 컬럼에만 결측치 존재
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].where(df[categorical_cols].notna(), 'Unknown')
        df['PGD 시술 여부'] = df['PGD 시술 여부'].fillna(0)  

        # 4. 도메인 적으로 처리한 결측치
        '''
        배아 이식 경과일 > 난자 혼합 경과일 
        동결 배아 사용 여부가  == 0 & 기증 배아 사용 여부 == 0 & 배아 이식 경과일이 0~6 -> 신선 배아 여부로 배아 이식한 것
        즉, 난자 혼합 경과일은 존재해야함. 결측치로 있으면 안됨. -> 일단 배아 이식 경과일보다 적은 값인 0으로 처리함
        하지만 0~6이 상황에 따라 될 수도 있는 가능성..
        '''
        condition = (
            (df['배아 이식 경과일'].between(0, 6)) &
            (df['동결 배아 사용 여부'] == 0) &
            (df['기증 배아 사용 여부'] == 0)
        )
        df.loc[condition, '난자 혼합 경과일'] = df.loc[condition, '난자 혼합 경과일'].fillna(0)
    
        return df

    ######################################################################################################


    # 3. 파생변수 생성

    def generate_features(df):

    ############ 파생 변수 생성을 위한 함수 ##############
        def convert_to_numeric(value):
            if value == '6회 이상':
                return 6
            try:
                return int(value.replace('회', ''))
            except:
                return 0  # 예외처리

        def calculate_total_treatment(row):
            ivf = row["IVF 시술 횟수"]
            di = row["DI 시술 횟수"]
        
            # 숫자 형태로 반환 (문자열 아님)
            if ivf >= 6 and di >= 6:
                return 12  # 둘 다 6회 이상이면 최소 12회로 가정
            elif ivf >= 6:
                return 6 + di
            elif di >= 6:
                return 6 + ivf
            else:
                return ivf + di  # 일반적인 경우 단순 합산
        
        def calculate_total_pregnancy(row): 
            ivf = row["IVF 임신 횟수"]
            di = row["DI 임신 횟수"]
            return ivf + di
        
        def calculate_total_birth(row):
            ivf = row["IVF 출산 횟수"]
            di = row["DI 출산 횟수"]
            return ivf + di
            
    
     ############# 파생 변수 생성 ##############

        # 1.
        numeric_cols = ["DI 시술 횟수", "DI 임신 횟수", "DI 출산 횟수", 
                        "IVF 시술 횟수", "IVF 임신 횟수", "IVF 출산 횟수",
                        '총 시술 횟수', '클리닉 내 총 시술 횟수']
    
        # 6회 이상 여부를 저장할 새로운 컬럼 생성 (일부 열 제외)
        for col in numeric_cols:
            df[col] = df[col].astype(str).apply(convert_to_numeric)

        # '총 시술 횟수' 수정 (IVF + DI 합산)
        df["총 시술 횟수"] = df.apply(calculate_total_treatment, axis=1)
    
        # '총 임신 횟수' 수정 (IVF 임신 횟수 + DI 임신 횟수)
        df["총 임신 횟수"] = df.apply(calculate_total_pregnancy, axis=1)
    
        # '총 출산 횟수' 수정 (IVF 출산 횟수 + DI 출산 횟수)
        df["총 출산 횟수"] = df.apply(calculate_total_birth, axis=1)
        
        


    

        
        df["IVF 임신 대비 출산율"] = df["IVF 출산 횟수"] / df["IVF 임신 횟수"]
        df["IVF 임신 대비 출산율"] = df["IVF 임신 대비 출산율"].fillna(0)

        df["DI 임신 대비 출산율"] = df["DI 출산 횟수"] / df["DI 임신 횟수"]
        df["DI 임신 대비 출산율"] = df["DI 임신 대비 출산율"].fillna(0)



        '''
        아래 파생 변수 약 25개 정도 만들었는데
        현재 코드에서 학습된 변수 조합해서 계속 추가하면서 여러 조합 짜면서 괜찮은 변수 조합 찾아내기
        기록은 노션에다 부탁드립니다.
        
        '''
        

        ###### 파생 변수 많이 추가해보기 ############################################################
        
        # 비율 값에서 무한대 및 NaN을 0으로 처리하는 함수(-1로 도전?)
        def safe_divide(numerator, denominator):
            return np.where(denominator == 0, 0, numerator / denominator)
        
        
        # 1. 출산 성공률
        df["출산 성공률"] = safe_divide(df["총 출산 횟수"], df["총 임신 횟수"])
        
        # 2. 저장된 배아 수 대비 이식된 배아 비율
        df["저장된 배아 수 대비 이식된 배아 비율"] = safe_divide(df["이식된 배아 수"], df["저장된 배아 수"])
        
        # 3. 해동 배아 비율
        df["해동 배아 비율"] = safe_divide(df["해동된 배아 수"], df["저장된 배아 수"])
        
        # 4. 미세주입 비율
        df["미세주입 비율"] = safe_divide(df["미세주입된 난자 수"], df["수집된 신선 난자 수"])
        
        # 5. 혼합된 난자 비율
        df["혼합된 난자 비율"] = safe_divide(df["혼합된 난자 수"], df["수집된 신선 난자 수"])
        
        # 6. 불임 원인 총합
        불임_변인_목록 = [
            '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', 
            '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증',
            '남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인',
            '부부 주 불임 원인', '부부 부 불임 원인'
        ]
        existing_columns = [col for col in 불임_변인_목록 if col in df.columns]
        df['불임 원인 총합'] = df[existing_columns].sum(axis=1)
        
        # 7. 배아 생성 대비 저장된 배아 비율
        df["배아 생성 대비 저장된 배아 비율"] = safe_divide(df["저장된 배아 수"], df["총 생성 배아 수"])
        
        # 8. 배아 생성 대비 해동된 배아 비율
        df["배아 생성 대비 해동된 배아 비율"] = safe_divide(df["해동된 배아 수"], df["총 생성 배아 수"])
        
        # 9. 버려진 배아 비율
        df["버려진 배아 비율"] = safe_divide(df["총 생성 배아 수"] - df["이식된 배아 수"] - df["저장된 배아 수"], df["총 생성 배아 수"])
        
        # 10. IVF 사용 빈도
        df["IVF 사용 빈도"] = safe_divide(df["IVF 시술 횟수"], df["총 시술 횟수"])
        
        # 11. DI 사용 빈도
        df["DI 사용 빈도"] = safe_divide(df["DI 시술 횟수"], df["총 시술 횟수"])
        
        # 12. 남성 불임 원인 총합
        df['남성 불임 원인 총합'] = df[['남성 주 불임 원인', '남성 부 불임 원인', '불임 원인 - 남성 요인']].sum(axis=1)
        
        # 13. 여성 불임 원인 총합
        df['여성 불임 원인 총합'] = df[['여성 주 불임 원인', '여성 부 불임 원인', '불임 원인 - 여성 요인',
                                          '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증']].sum(axis=1)
        
        # 14. 부부 불임 원인 총합
        df['부부 불임 원인 총합'] = df[['부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인']].sum(axis=1)
        
        # 15. 정자 문제 종합
        df['정자 문제 종합'] = df[['불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인',
                                    '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']].sum(axis=1)
        
        # 16. 환자의 현재 자궁 상태
        def categorize_uterus_condition(row):
            if row["총 생성 배아 수"] != 0 and row["저장된 배아 수"] != 0:
                if row["이식된 배아 수"] == 0:
                    return "좋지 않음"
                else:
                    return "좋음"
            return "알 수 없음"
        
        df["환자의 현재 자궁 상태"] = df.apply(categorize_uterus_condition, axis=1)
        
        # 17. 자연 인공 혼합 생성 배아 수
        df["자연 인공 혼합 생성 배아 수"] = df["총 생성 배아 수"] - df["미세주입에서 생성된 배아 수"] - df["저장된 배아 수"]
        
        # 18. 저장된 자연 인공 혼합 배아 수
        df["저장된 자연 인공 혼합 배아 수"] = df["저장된 배아 수"] - df["미세주입 후 저장된 배아 수"]
        
        # 19. 자연 인공 혼합 배아 이식 수
        mask = (df["동결 배아 사용 여부"] == 0) & (df["기증 배아 사용 여부"] == 0)
        df.loc[mask, "자연 인공 혼합 배아 이식 수"] = df["이식된 배아 수"] - df["미세주입 배아 이식 수"]
        
        # 20. 생성한 배아 이식 수, 생성된 품질 좋은 배아 수
        df.loc[mask, "생성한 배아 이식 수"] = df["미세주입 배아 이식 수"] + df["자연 인공 혼합 배아 이식 수"]
        df.loc[mask, "생성된 품질 좋은 배아 수"] = df["저장된 배아 수"] + df["생성한 배아 이식 수"]
        df.loc[~mask, ["자연 인공 혼합 배아 이식 수", "생성한 배아 이식 수", "생성된 품질 좋은 배아 수"]] = np.nan
        
        # 21. 유산 횟수
        df["유산 횟수"] = df["총 임신 횟수"] - df["총 출산 횟수"]


        # 22.저장된 배아 수(위험도)
        def categorize_stored_embryo_count(n):
            if 0 <n <= 6:
                return "최적 범위"
            elif 6 < n <= 12:
                return "저위험"
            elif n > 12:
                return "고위험"
            else:
                return "알 수 없음"
        df["저장된 배아 수(위험도)"] = df["저장된 배아 수"].apply(categorize_stored_embryo_count)

        # 23.
        # 새로운 범주로 저장된 배아 수(위험도) 분류하는 함수 정의
        def categorize_mixed_egg_count(n):
            if 0 < n <= 3 or 30 <= n <= 40:
                return "고위험"
            elif 4 <= n <= 7 or 25 <= n <= 29:
                return "저위험"
            elif 8 <= n <= 24:
                return "적정"
            else:
                return "알 수 없음"

        # 기존 데이터프레임이 있다고 가정하고 새로운 범주 열 추가
        df["혼합된 난자 수(위험도)"] = df["혼합된 난자 수"].apply(categorize_mixed_egg_count)

        ##################################################################################################################






        
        # 2. 재현을 위한 각주 --> 사실 더 나은 전처리는 각주를 풀어야함..
        # df["특정 시술 유형"] = df["특정 시술 유형"].astype(str).str.strip()
        # df[["시술 조합 1", "시술 조합 2"]] = df["특정 시술 유형"].str.split(":", n=1, expand=True)
        # df["시술 조합 2"] = df["시술 조합 2"].fillna("")
        # df["시술 조합 1"] = df["시술 조합 1"].str.strip()
        # df["시술 조합 2"] = df["시술 조합 2"].str.strip()


        
        # 3.
        # 배양 일 수 범주화
        def categorize_culture_days(days):
            if days < 0:  
                return "알 수 없음"
            elif days == 3:
                return "3일 배양"
            elif days == 5:
                return "5일 배양"
            elif days in [1, 2]:  
                return "3일 배양"
            elif days == 4:  
                return "5일 배양"
            elif days in [6, 7]:  
                return "5일 배양"
            else:
                return "알 수 없음"

        df["신선 배아 배양 일 수"] = (df["배아 이식 경과일"] - df["난자 혼합 경과일"])
        df["신선 배아 배양 유형"] = df["신선 배아 배양 일 수"].apply(categorize_culture_days)
        
        df["동결 배아 배양 일 수"] = (df["배아 이식 경과일"] - df["난자 해동 경과일"])
        df["동결 배아 배양 유형"] = df["동결 배아 배양 일 수"].apply(categorize_culture_days)
        
        df["배아 배양 유형"] = np.where(
            df["동결 배아 배양 유형"] != "알 수 없음",
            df["동결 배아 배양 유형"],
            df["신선 배아 배양 유형"]
        )


        # 4.
        df["현재 시술용 여부"] = df["배아 생성 주요 이유"].apply(lambda x: "1" if isinstance(x, str) and "현재 시술용" in x.split(",") else "0")


        
        # 5.
        # 배아 이식 대비 생성 비율 파생변수
        def categorize_embryo_implantation_rate(rate):
            if rate > 100:
                return "알 수 없음"  # 100% 초과 데이터 처리
            elif 30 <= rate <= 100:
                return "1"
            elif 0 < rate < 30:
                return "0"
            else:  # rate == 0
                return "알 수 없음"

        df['배아 이식 대비 생성 비율'] = (df['이식된 배아 수'] / df['총 생성 배아 수']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        df['배아 이식 대비 생성 비율'] = df['배아 이식 대비 생성 비율'].astype(int)
        df['배아 이식 대비 생성 비율 분류'] = df['배아 이식 대비 생성 비율'].apply(categorize_embryo_implantation_rate)
        df = df.drop(columns='배아 이식 대비 생성 비율')
        # df.loc[df['시술 조합 1'].str.contains('BLASTOCYST', na=False) | df['시술 조합 2'].str.contains('BLASTOCYST', na=False), '배아 배양 유형'] = '5일 배양'




        
        # 추가 2
        # 배아 품질 위험 지수 (생성 배아 수 기준)
        def categorize_embryo_risk(total_embryos):
            if total_embryos < 5:
                return "고위험"
            else: 
                return "중간 위험"
        df['배아 품질 위험 지수(생성 배아 수 기준)'] = df['총 생성 배아 수'].apply(categorize_embryo_risk)


        
        def categorize_egg_count(n):
            if n < 10:
                return "고위험 (10개 미만)"
            elif 10 <= n <= 15:
                return "최적 범위 (10~15개)"
            elif 16 <= n <= 20:
                return "저위험 (16~20개)"
            else:
                return "중위험 (20개 초과)"
        
        df["난자 채취 범주"] = df["수집된 신선 난자 수"].apply(categorize_egg_count)
        
        #===================================================================================================================
        # PGS, PGD 유전진단 여부
        # PGS_검사진단 열 생성
        df["PGS_검사진단"] = (
            df[["착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부", "PGS 시술 여부"]]
            .fillna(0)  # NaN을 0으로 채움
            .sum(axis=1) # 행 방향으로 합산
        )

        # PGD_검사진단 열 생성
        df["PGD_검사진단"] = (
            df[["착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부", "PGD 시술 여부"]]
            .fillna(0)
            .sum(axis=1)
        )

        # df['병원 이동 여부'] = ((df['총 시술 횟수'] > df['클리닉 내 총 시술 횟수']) & 
        #               (df['클리닉 내 총 시술 횟수'] != 0)).astype(int)
        # df["IVF_정자혼합"] = ((df["파트너 정자와 혼합된 난자 수"] == 0) & (df["기증자 정자와 혼합된 난자 수"] == 0) & (df["시술 유형"] == "IVF")).astype(int)

# "Blastocyst"가 포함된 경우 배아 배양 유형을 "5일 배양"으로 지정 (없으면 NaN 또는 다른 값 유지)    
        df.loc[df['특정 시술 유형'].str.contains('BLASTOCYST', case=False, na=False), '배아 배양 유형'] = '5일 배양'

        df["is_5day_culture"] = (df["배아 배양 유형"] == "5일 배양").astype(int)
        

        # 예시) PGS 사용 여부
        # PGS_검사진단이 0이면 미사용, 1 이상이면 사용(1)로 단순화
        df["is_pgs"] = (df["PGS_검사진단"] > 0).astype(int)

        # 예시) PGD 사용 여부
        df["is_pgd"] = (df["PGD_검사진단"] > 0).astype(int)

        # 예시) 단일 배아 이식 여부는 이미 0/1로 존재
        # df["단일 배아 이식 여부"] = df["단일 배아 이식 여부"] (이미 0/1)

        # 예시) 35세 이하 여부
        # 시술 당시 나이가 '만18-34세'이면 1, 그 외는 0
        df["is_under_35"] = (df["시술 당시 나이"] == "만18-34세").astype(int)

        # 종합 파생 변수 (원하시는 형태로 계산 가능)
        # 예) 5개 조건(5일배양, PGS, PGD, 단일배아이식, 35세 이하)을 모두 합산
        df["combined_feature"] = (
            df["is_5day_culture"] 
            + df["is_pgs"] 
            + df["is_pgd"] 
            + df["단일 배아 이식 여부"] 
            + df["is_under_35"]
        )
        mask = (
            (df["특정 시술 유형"] == "Unknown") &
            (
                (df["미세주입된 난자 수"] != 0) |
                (df["미세주입에서 생성된 배아 수"] != 0) |
                (df["미세주입 배아 이식 수"] != 0) |
                (df["미세주입 후 저장된 배아 수"] != 0)
            )
        )

        df.loc[mask, "특정 시술 유형"] = "ICSI"



        
        def update_treatment_and_create_sperm_factor(df):

            
            # 불임 원인 - 정자 농도/면역학적 요인/운동성/형태를 모두 더한 컬럼 '불임원인-정자' 생성
            df['불임원인-정자'] = (
                df['불임 원인 - 정자 농도'] +
                df['불임 원인 - 정자 면역학적 요인'] +
                df['불임 원인 - 정자 운동성'] +
                df['불임 원인 - 정자 형태']
            )
            
            return df

        df = update_treatment_and_create_sperm_factor(df)


    
        print("\n📌 [파생 변수 생성 완료]\n")
        return df


    # 학습하고 있는 변수 출력하는 함수
    def print_current_features(df):
        """
        드랍된 후 남아 있는 학습 변수 목록을 간단히 출력하는 함수
    
        Parameters:
        - df: 드랍이 완료된 후의 데이터프레임
    
        Returns:
        - None (변수 목록만 출력)
        """
        print("\n🔹 현재 학습에 사용되는 변수 목록:")
        print(f"총 {df.shape[1]}개 변수 사용 중")
        print(", ".join(df.columns))  # 변수 이름만 출력


        

 
    ## 함수 main
    df = preprocess_missing_values(df) # 결측치 처리
    df = remove_invalid_rows(df, is_train = is_train) # 오류 행 제거
    df = generate_features(df) # 컬럼 분해 및 파생 변수 생성
    df.drop(columns=drop_list, errors='ignore', inplace=True) # 컬럼 제거
    print_current_features(df)
    cols_to_convert = [col for col in df.columns if col != '임신 성공 여부'] # 모든 열 타입 object로
    df = df.astype({col: str for col in cols_to_convert})
        

    return df










# 결과 시각화
def Result_visualize(auc_list, feature_importance_list, x_train, top_features=100):
    """ 
    F1 스코어 출력 및 특징 중요도 시각화 함수

    Parameters:
    - auc_list (list): AUC 점수 리스트
    - feature_importance_list (list of arrays): 모델별 특징 중요도 리스트
    - x_train (DataFrame): 학습 데이터셋 (특징 이름을 가져오기 위함)
    - top_features (int): 상위 N개의 특징만 시각화 (기본값 100)
    
    Returns:
    - mean_auc (float): 평균 AUC 점수
    - sorted_feature_names (list): 중요도 기준 정렬된 상위 특징 이름 리스트
    - sorted_mean_importance (array): 정렬된 특징 중요도 배열
    """

    # 평균 AUC 점수 출력
    mean_auc = np.mean(auc_list)
    print("예상 Score:", mean_auc)

    # 특징 중요도 평균 계산 및 정렬
    mean_importance = np.mean(feature_importance_list, axis=0)
    sorted_indices = np.argsort(mean_importance)[::-1][:top_features]  # 오름차순 정렬 후 top_features 선택
    sorted_feature_names = [x_train.columns[i] for i in sorted_indices]
    sorted_mean_importance = mean_importance[sorted_indices]

    # 중요도 시각화
    plt.figure(figsize=(10, 20))
    plt.barh(sorted_feature_names, sorted_mean_importance)
    plt.xlabel('Mean Importance')
    plt.ylabel('Features')
    plt.title('Mean Feature Importance')
    plt.show()

    return mean_auc, sorted_feature_names, sorted_mean_importance


# 평가 지표 뽑아보기
def get_clf_eval(y_test, y_pred=None, y_pred_probs=None):
    # 오차 행렬 계산
    confusion = confusion_matrix(y_test, y_pred)
    
    # 분류 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)  # 성공(1) 기준
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # AUC 계산 (확률 값 필요)
    if y_pred_probs is not None:
        auc = roc_auc_score(y_test, y_pred_probs)
        print("ROC AUC: {:.4f}".format(auc))
    else:
        auc = None  # 확률 값이 없을 경우 None 반환

    # 출력
    print("오차 행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))
    
    return accuracy, precision, recall, f1, auc







import os
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from yong import Preprocessing, Result_visualize, get_clf_eval
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


train_data = pd.read_csv('./train.csv').drop(columns=["ID"], errors="ignore")
test_data = pd.read_csv('./test.csv').drop(columns=["ID"], errors="ignore")

drop_list = ['불임 원인 - 여성 요인',

             '불임 원인 - 정자 면역학적 요인',
             '불임 원인 - 자궁경부 문제',
             '특정 시술 유형',
             '시술 조합 2',
             '배아 배양 일 수',
             #'배아 이식 대비 생성 비율 분류',
             #'배아 품질 위험 지수(생성 배아 수 기준)',
             'DI 임신 대비 출산율',
             'DI 시술 횟수_6회 이상 여부',
             'IVF 시술 횟수_6회 이상 여부',
             'IVF 임신 횟수_6회 이상 여부',
             '출산 성공률',
             '미세주입 비율',
             '불임 원인 총합',
             '유산 횟수',
             '배아 생성 대비 해동된 배아 비율',
             '혼합된 난자 비율',
             'PGD 시술 여부',
              'PGS 시술 여부',
              '난자 해동 경과일',
               '난자 혼합 경과일',
              '남성 주 불임 원인',
               '남성 부 불임 원인',
               '여성 주 불임 원인',
               '여성 부 불임 원인',
               '부부 주 불임 원인',
               '부부 부 불임 원인',
            #  '착상 전 유전 검사 사용 여부', '착상 전 유전 진단 사용 여부',
            # '동결 배아 사용 여부', '신선 배아 사용 여부', '기증 배아 사용 여부', '대리모 여부', 
            #   'PGS_검사진단',
               # 'PGD_검사진단',
               'is_5day_culture',
               'is_pgs',
               'is_pgd',
               'is_under_35',
               # 'combined_feature',
               # '불임원인-정자',
            #   '신선 배아 배양 일 수',
               # '신선 배아 배양 유형',
            #   '동결 배아 배양 일 수',
               # '동결 배아 배양 유형' 
               ]


RANDOM_STATE = 42

best_params = {
 'bagging_temperature': 0.34855471754199147,
 'class_weight_1': 1.0113851477295377,
 'depth': 7,
 'l2_leaf_reg': 3.7383117174881337,
 'learning_rate': 0.06239906179892577,
 'random_strength': 0.012175171854050976}


params = {
    "iterations": 2000,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "border_count": 128,
    "od_type": "Iter",
    "od_wait": 100,
    "use_best_model": True,
    "task_type": "GPU",
    "random_seed": RANDOM_STATE,
    "verbose": 100,

    # Optuna로 찾은 하이퍼파라미터 적용
    "learning_rate": best_params["learning_rate"],
    "depth": best_params["depth"],
    "l2_leaf_reg": best_params["l2_leaf_reg"],
    "bagging_temperature": best_params["bagging_temperature"],
    "random_strength": best_params["random_strength"],
    "class_weights": {0: 1.0, 1: best_params["class_weight_1"]},
}
N_SPLITS =10

# 데이터 프레임 전처리
train_data = Preprocessing(train_data, drop_list, is_train = True)
test_data = Preprocessing(test_data, drop_list, is_train = False)


cat_features = [col for col in train_data.columns if col != '임신 성공 여부']


# Prepare OOF array
oof_preds = np.zeros(len(train_data))
test_preds = np.zeros(len(test_data))  # For test-time averaging
# Store metrics for each fold
accuracy_list, precision_list, recall_list, f1_list, auc_list, feature_importance_list = [], [], [], [], [], []

# Stratified K-Fold
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data, train_data["임신 성공 여부"])):
    print(f"=============Training Fold {fold+1}...=============")
    
    # 데이터 분할
    df_train, df_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
    x_train, y_train = df_train.drop('임신 성공 여부', axis=1), df_train["임신 성공 여부"]
    x_val, y_val = df_val.drop('임신 성공 여부', axis=1), df_val["임신 성공 여부"]
    
    # Pool 생성
    train_pool = Pool(x_train, label=y_train, cat_features=cat_features)
    eval_pool = Pool(x_val, label=y_val, cat_features=cat_features)
    
    # CatBoost 모델 학습
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=eval_pool, verbose=100, early_stopping_rounds=250)
    
    # 각 폴드별 모델 가중치(모델 파라미터) 저장
    model_file_name = f"catboost_modelfor203_fold_{fold+1}.cbm"
    model.save_model(model_file_name)
    print(f"Fold {fold+1} 모델 저장 완료: {model_file_name}")
    
    # 예측 및 평가 진행
    val_pred_probs = model.predict_proba(x_val)[:, 1]
    val_pred = (val_pred_probs > 0.5).astype(int)
    oof_preds[val_idx] = val_pred_probs  
    
    accuracy, precision, recall, F1, auc = get_clf_eval(y_val, val_pred, val_pred_probs)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(F1)
    auc_list.append(auc)
    feature_importance_list.append(model.get_feature_importance())
    
    test_preds += model.predict_proba(test_data)[:, 1] / N_SPLITS
    
    
    def print_metrics(metrics):
    print("\n📌 Final Cross-Validation Results:")
    for name, values in metrics.items():
        print(f"   - {name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


metrics = {
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1 Score": f1_list,
    "AUC": auc_list
}

print_metrics(metrics)

''' 
📌 Final Cross-Validation Results:
   - Accuracy: 0.7463 ± 0.0010
   - Precision: 0.5371 ± 0.0081
   - Recall: 0.1311 ± 0.0055
   - F1 Score: 0.2107 ± 0.0072
   - AUC: 0.7405 ± 0.0027


'''