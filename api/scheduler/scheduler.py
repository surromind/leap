import os
import json
import signal
import requests
import datetime
from typing import Any, Dict, Optional
from multiprocessing import Queue, Manager


from .utils import get_gputil_info
from config.url import TYPE_URL


class Scheduler:
    def __init__(self):
        """
        작업 스케줄링을 담당하는 클래스
        """
        manager = Manager()
        self.request_queue = Queue()  # 작업 요청 큐
        self.result_queue = Queue()  # 작업 결과 큐
        self.tasks_status = manager.dict()  # 작업 상태 정보 저장

    def get_status(self) -> Dict[str, Any]:
        """
        gpu 메모리 사용량 및 큐 상태 확인
        """
        status_dict = get_gputil_info()  # GPU 정보 추가
        status_dict.update(self.tasks_status)  # 작업 상태 정보 추가
        return status_dict

    def get_current_task(self) -> Optional[str]:
        """
        현재 진행중인 작업 확인
        """
        for task_id, task_info in self.tasks_status.items():
            if task_info["status"] == "running":
                return task_id
        return None

    def stop_task(self) -> Dict[str, Any]:
        """
        현재 진행중인 작업 확인 후 중지
        """
        current_task = self.get_current_task()  # 현재 진행중인 작업 확인
        if current_task == None:
            return {"error": "task is not running"}
        else:
            info = self.tasks_status[current_task]
            os.kill(info["process"], signal.SIGTERM)  # 해당 작업 프로세스 종료
            self.tasks_status[current_task] = {
                "status": "stopped",
                "args": info["args"],
            }  # 작업 상태 업데이트
            return {"id": current_task, "status": "stopped"}

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        대기 중인 작업 취소
        """
        # 작업이 대기 중이 아닌 경우
        if task_id not in self.tasks_status.keys():
            return {"error": "task is not in queue"}
        # 작업이 대기 중인 경우
        else:
            # 작업 큐에서 해당 작업 제거
            while not self.request_queue.empty():
                get_type, get_id, get_args = self.request_queue.get()
                if get_id != task_id:
                    self.request_queue.put((get_type, get_id, get_args))
            self.tasks_status.pop(task_id)
            return {"id": task_id, "status": "canceled"}

    def add_task(self, type: str, args: Dict[str, Any]):
        """
        평가 Arguments를 작업 요청 큐에 추가
        """
        now = datetime.datetime.now()
        model_name = args["model"].split("/")[-1]
        task_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{model_name}"  # 작업 ID 생성
        self.request_queue.put((type, task_id, args))  # 작업 요청 큐에 추가
        self.tasks_status[task_id] = {
            "type": type,
            "status": "queued",
            "args": args,
        }  # 작업 상태 업데이트
        return {"id": task_id, "status": "queued"}

    def manage_tasks(self):
        """
        내부 큐 관리 프로세스가 수행하는 함수
        """
        # 서버 실행되는 동안 계속 실행
        while True:
            # 현재 진행중인 작업이 없는 경우 작업 시작
            if not self.get_current_task():
                # 작업 요청 큐에서 작업 가져오기
                type, task_id, args = self.request_queue.get()
                input = {"task_id": task_id, "args": args}

                # 작업 프로세스 생성 및 시작, 완료 대기
                self.tasks_status[task_id] = {
                    "type": type,
                    "status": "running",
                    "args": args,
                }  # 작업 상태 업데이트
                result = requests.post(
                    TYPE_URL[type],
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(input),
                )
                # 작업 서버가 비정상(500 등, non-JSON) 응답을 줘도
                # 스케줄러 프로세스가 죽지 않도록 방어
                try:
                    result = result.json()
                except Exception:
                    result = {
                        "error": f"HTTP {result.status_code}",
                        "detail": result.text[:1000],
                    }
                self.result_queue.put((task_id, result))

            # 진행 중이었던 작업이 완료된 경우
            else:
                # 작업 결과 큐에서 결과 가져오기
                task_id, result = self.result_queue.get()
                self.tasks_status[task_id] = {
                    "type": self.tasks_status[task_id]["type"],
                    "status": "completed",
                    "args": self.tasks_status[task_id]["args"],
                    "result": result,
                }  # 작업 상태 업데이트
