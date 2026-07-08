import os
import pandas as pd
from .path import MODEL_INFO_PATH, SAVE_MODEL_DIR, MODEL_DIR


def _mark_local(model_name):
    """
    로컬(MODEL_DIR)에 완전히 받아져 있는 모델은 표시 라벨에 🟢 마커를 붙여
    시연 시 다운로드 없이 바로 쓸 수 있는 모델을 구분한다.
    Gradio Dropdown 의 [표시라벨, 실제값] 형식으로 반환하므로 실제 값(모델명)은 그대로 유지된다.
    """
    is_local = os.path.isfile(os.path.join(MODEL_DIR, model_name, "config.json"))
    label = f"🟢 {model_name}" if is_local else model_name
    return [label, model_name]


def _marked_sorted(model_names):
    """
    모델명 리스트를 (표시라벨, 값) 쌍으로 마킹하고, 로컬(🟢) 모델이 목록 상단에 오도록 정렬한다.
    stable sort 라 로컬/비로컬 각 그룹 내부의 원래 순서는 유지된다.
    """
    pairs = [_mark_local(m) for m in model_names]
    pairs.sort(key=lambda p: 0 if p[0].startswith("🟢") else 1)
    return pairs


def list_folders(directory):
    """
    주어진 디렉토리 아래의 모든 폴더 이름을 반환합니다.

    Args:
        directory (str): 검색할 디렉토리 경로

    Returns:
        list: 폴더 이름의 리스트
    """
    folders = []
    try:
        for item in os.listdir(directory):
            if item.endswith("_Merge"):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    folders.extend(
                        [
                            os.path.join(
                                "llm-tuning-model", item_path.split("/")[-1], checkpoint
                            )
                            for checkpoint in os.listdir(item_path)
                        ]
                    )
    except:
        pass
    return folders


def get_eval_models():
    model_info = pd.read_json(MODEL_INFO_PATH)
    EVAL_MODELS = {}
    EVAL_MODELS["baseline"] = _marked_sorted(list(model_info["Model"]))
    EVAL_MODELS["finetuned"] = _marked_sorted(
        list_folders(SAVE_MODEL_DIR)
    )  # 튜닝 모델을 평가 모델로 추가
    return EVAL_MODELS


def get_tuning_models():
    TUNING_MODELS = {
        "tuning": _marked_sorted(
            [
                "Qwen/Qwen2.5-1.5B",
                "Qwen/Qwen2.5-3B",
                "Qwen/Qwen2.5-7B",
                "Qwen/Qwen2.5-14B",
                "meta-llama/Llama-3.1-8B",
                "google/gemma-1.1-2b-it",
                "google/gemma-1.1-7b-it",
                "google/gemma-2-2b",
                "google/gemma-2-9b",
            ]
        )
    }
    return TUNING_MODELS
