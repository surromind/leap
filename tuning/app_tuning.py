import os
import sys
import json
import shutil
import uvicorn
import subprocess

from fastapi import FastAPI, Request
from fastapi.responses import Response


app = FastAPI()

from config.train import PER_STEP_ARG, PER_STEP_CMD
from config.logger import LOG_DIR


def get_command(step, args, device):
    """입력 데이터에서 명령어로 변환"""
    # merge 명령어에 checkpoint 붙여주기
    if step.startswith("run_merge"):
        checkpoints = [
            name
            for name in os.listdir(args["output_dir"])
            if name.startswith("checkpoint")
        ]
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        checkpoint = checkpoints[int(step.split("_")[-1])]
        step = "_".join(step.split("_")[:-1])
        args["adapter_path"] = f"{args['output_dir']}/{checkpoint}"
        args["merge_output_dir"] = f"{args['output_dir']}_Merge/{checkpoint}"

    # command 인자 붙이기
    command = list(PER_STEP_CMD[step])

    for arg_name in PER_STEP_ARG[step]:
        value = args[arg_name]
        command.extend([f"--{arg_name}", str(value)])

    # device 개수 수정
    if step == "run_tuning":
        command[2] = str(len(device))

    return command


@app.post("/tuning")
async def request(request: Request) -> Response:
    # 입력 데이터 추출
    input = await request.json()
    args = input["args"]
    task_id = input["task_id"]

    # device 설정
    device = args.pop("device", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device)
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # 베이스 모델이 로컬에 없으면 허브에서 자동 다운로드 (seaweedfs 폐기로 모델이 미리
    # 채워져 있지 않을 수 있음). 없을 경우 AutoTokenizer.from_pretrained 가 로컬 경로를
    # 허브 repo_id 로 오해해 HFValidationError 로 죽는다. model_dir 이 args 에 없으므로
    # 경로의 마지막 두 요소(namespace/model)로 repo_id 를 도출한다.
    model_path = args.get("model")
    if model_path and not os.path.exists(model_path):
        repo_id = "/".join(model_path.rstrip("/").split("/")[-2:])
        try:
            from huggingface_hub import snapshot_download

            print(f"[tuning] base model 로컬에 없음 -> 허브 다운로드: {repo_id} -> {model_path}")
            snapshot_download(repo_id=repo_id, local_dir=model_path)
            print(f"[tuning] base model 다운로드 완료: {model_path}")
        except Exception as e:
            print(f"[tuning] base model 다운로드 실패({repo_id}): {e}")

    # STEP 처리
    steps = ["convert_datasets", "run_tuning"]
    steps.extend([f"run_merge_{idx}" for idx in range(int(args["num_train_epochs"]))])

    # STEP별로 실행
    result = {step: "not running" for step in steps}

    with open(f"{LOG_DIR}/{task_id}.log", "a") as log_file:
        sys.stdout = log_file  # 표준 출력 재지정
        sys.stderr = log_file  # 표준 에러 재지정
        try:
            for step in steps:
                # command 처리
                command = get_command(step, args, device)
                # 로그 기록
                print(f"Step: {step}")
                print(f"Running command: {' '.join(command)}")

                # 명령어 실행
                try:
                    subprocess.run(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    result[step] = "success"
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e}")
                    result[step] = str(e)
                    break
        finally:
            # 표준 출력 및 에러 복원
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            # 데이터셋 변환 임시 폴더 삭제
            if os.path.exists(args["dataset_path"]):
                if os.path.isdir(args["dataset_path"]):
                    try:
                        shutil.rmtree(args["dataset_path"])
                    except:
                        pass

    return Response(
        content=json.dumps(dict(result), indent=4), media_type="application/json"
    )


if __name__ == "__main__":
    # 서버 실행
    api_host = "0.0.0.0"
    api_port = int(os.getenv("TUNING_PORT", "11182"))
    uvicorn.run(app, host=api_host, port=api_port)
