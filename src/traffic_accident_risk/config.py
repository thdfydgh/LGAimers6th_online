"""Static configuration for training."""

from __future__ import annotations

from typing import Dict, List


TARGET_COL = "임신 성공 여부"
ID_COL = "ID"

DROP_COLUMNS: List[str] = [
    "불임 원인 - 여성 요인",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 자궁경부 문제",
    "특정 시술 유형",
    "시술 조합 2",
    "배아 배양 일 수",
    "DI 임신 대비 출산율",
    "DI 시술 횟수_6회 이상 여부",
    "IVF 시술 횟수_6회 이상 여부",
    "IVF 임신 횟수_6회 이상 여부",
    "출산 성공률",
    "미세주입 비율",
    "불임 원인 총합",
    "유산 횟수",
    "배아 생성 대비 해동된 배아 비율",
    "혼합된 난자 비율",
    "PGD 시술 여부",
    "PGS 시술 여부",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "is_5day_culture",
    "is_pgs",
    "is_pgd",
    "is_under_35",
]

BEST_PARAMS = {
    "bagging_temperature": 0.34855471754199147,
    "class_weight_1": 1.0113851477295377,
    "depth": 7,
    "l2_leaf_reg": 3.7383117174881337,
    "learning_rate": 0.06239906179892577,
    "random_strength": 0.012175171854050976,
}


def get_catboost_params(task_type: str = "GPU", random_state: int = 42) -> Dict[str, object]:
    return {
        "iterations": 2000,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "border_count": 128,
        "od_type": "Iter",
        "od_wait": 100,
        "use_best_model": True,
        "task_type": task_type,
        "random_seed": random_state,
        "verbose": 100,
        "learning_rate": BEST_PARAMS["learning_rate"],
        "depth": BEST_PARAMS["depth"],
        "l2_leaf_reg": BEST_PARAMS["l2_leaf_reg"],
        "bagging_temperature": BEST_PARAMS["bagging_temperature"],
        "random_strength": BEST_PARAMS["random_strength"],
        "class_weights": {0: 1.0, 1: BEST_PARAMS["class_weight_1"]},
    }

