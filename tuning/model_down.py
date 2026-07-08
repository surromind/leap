import os
from multiprocessing import Process, Queue
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download, login


def hf_down(repo, save_path):
    snapshot_download(repo_id=repo, local_dir=save_path, local_dir_use_symlinks=False)


if __name__ == "__main__":
    # HF_TOKEN 환경변수를 우선 사용 (docker-compose .env 로 주입). 없으면 기존 하드코딩 값 fallback.
    login(token=os.getenv("HF_TOKEN", "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"))
    save_home_dir = "/home/leap/"
    processes = [
        Process(
            target=hf_down,
            args=(
                "google/gemma-2-2b-it",
                save_home_dir + "gemma-2-2b-it",
            ),
        ),
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
