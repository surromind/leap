import os
import json
import random
import logging
from typing import Literal
from multiprocessing import Process

import uvicorn
from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response

from scheduler import Scheduler
from config.path import (
    LEADERBOARD_PATH,
    MODEL_INFO_PATH,
    EVAL_RESULTS_PATH,
    SAVE_DATASET_DIR,
    ensure_paths_exist,
)
from config import (
    get_eval_models,
    get_tuning_models,
    get_eval_datasets,
    get_tuning_datasets,
)

app = FastAPI()


class ExcludePathFilter(logging.Filter):
    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = set(paths)  # 리스트를 집합으로 변환 (빠른 조회를 위해)

    def filter(self, record: logging.LogRecord) -> bool:
        # 로그 메시지에 제외 경로가 포함되어 있는지 확인
        return not any(path in record.getMessage() for path in self.paths)


# 제외할 경로들
exclude_paths = ["/status", "/list"]

# 필터 적용
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(ExcludePathFilter(exclude_paths))


# 테스트 요청
@app.get("/")
async def test():
    return "hello world"


# gpu 메모리 사용량 및 큐 상태 확인
@app.get("/status")
async def get_status():
    return scheduler.get_status()


# 현재 진행중인 작업 확인
@app.get("/current_task")
async def get_current_task():
    return scheduler.get_current_task()


# 평가 요청 수신
@app.post("/evaluate")
async def add_evaluate_request(request: Request) -> Response:
    args = await request.json()
    response = scheduler.add_task("evaluate", args)
    return JSONResponse(response)


@app.post("/tuning")
async def add_tuning_request(request: Request) -> Response:
    args = await request.json()
    response = scheduler.add_task("tuning", args)
    return JSONResponse(response)


# 대기 중인 특정 요청 취소
@app.post("/cancel/{task_id}")
async def cancel(task_id: str):
    response = scheduler.cancel_task(task_id)
    return JSONResponse(response)


@app.get("/data/{data_type}")
async def get_data(data_type: Literal["leaderboard", "model_info", "eval_results"]):
    per_type_path = {
        "leaderboard": LEADERBOARD_PATH,
        "model_info": MODEL_INFO_PATH,
        "eval_results": EVAL_RESULTS_PATH,
    }
    with open(per_type_path[data_type], "rb") as f:
        return Response(
            f.read(),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=leaderboard.json"},
        )


@app.post("/data/{data_type}")
async def save_data(
    data_type: Literal["leaderboard", "model_info", "eval_results"],
    request: Request,
):
    data = await request.json()
    per_type_path = {
        "leaderboard": LEADERBOARD_PATH,
        "model_info": MODEL_INFO_PATH,
        "eval_results": EVAL_RESULTS_PATH,
    }
    with open(per_type_path[data_type], "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    return Response(status_code=200)


@app.get("/list/{task_type}/{list_type}")
async def get_list(
    task_type: Literal["eval", "tuning"], list_type: Literal["model", "dataset"]
):
    per_type_list = {
        "eval": {"model": get_eval_models(), "dataset": get_eval_datasets()},
        "tuning": {"model": get_tuning_models(), "dataset": get_tuning_datasets()},
    }

    return per_type_list[task_type][list_type]


@app.get("/dataset/sample/{dataset_name}")
async def get_data_sample(dataset_name: str):
    dataset_name = dataset_name.replace("$", "/")
    # 존재하지 않는 데이터셋명이 와도 500(KeyError) 대신 빈 샘플을 반환해
    # 프론트엔드의 response.json() 이 깨지지 않도록 한다.
    file_path = get_tuning_datasets().get(dataset_name)
    if not file_path or not os.path.exists(file_path):
        return {"sample": [], "error": f"dataset not found: {dataset_name}"}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            data = json.load(file)
    except Exception as e:
        return {"sample": [], "error": f"failed to read dataset: {e}"}
    if isinstance(data, list):
        random.shuffle(data)
        return {"sample": data[0:3]}
    return {"sample": data}


@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile):
    try:
        with open(os.path.join(SAVE_DATASET_DIR, file.filename), "wb") as f:
            content = await file.read()
            f.write(content)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 경로 확인
    ensure_paths_exist()
    # 스케줄러 초기화
    scheduler = Scheduler()
    # 내부 큐 관리 프로세스 생성
    Process(target=scheduler.manage_tasks).start()
    # 서버 실행
    api_host = "0.0.0.0"
    api_port = int(os.getenv("BACKEND_PORT", "11181"))
    uvicorn.run(app, host=api_host, port=api_port)
