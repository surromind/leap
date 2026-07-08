import os
import shutil
import collections
from multiprocessing import Process
from huggingface_hub import snapshot_download, login


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr:
        yield arr


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0] for y in x], x[0][1]) for x in arr]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


def hf_down(repo, save_path):
    snapshot_download(repo_id=repo, local_dir=save_path, local_dir_use_symlinks=False)


def download_model(model_dir, model_name):
    path = os.path.join(model_dir, model_name)
    try:
        # HF_TOKEN 이 유효하지 않아도 공개 모델은 익명으로 다운로드 가능하므로 login 실패는 무시한다.
        # (예전엔 무효 토큰으로 login 이 예외를 던져 공개 모델 다운로드까지 막았다. gated 모델을
        #  받으려면 유효한 HF_TOKEN 을 환경변수/Dockerfile 에 설정해야 한다.)
        token = os.getenv("HF_TOKEN")
        if token:
            try:
                login(token=token)
            except Exception as login_err:
                print(f"HF login 실패, 익명 다운로드로 계속 진행: {login_err}")
        # NOTE: 예전에는 hf_down 을 별도 Process 로 감쌌으나, 이 함수 자체가 이미
        # eval 서브프로세스(Process(target=eval)) 안에서 호출되어 중첩 fork 로
        # 다운로드가 조용히 실패했고, 부모는 실패를 삼킨 채 넘어가 vLLM 이 빈 경로를
        # 로드하다 "Can't load configuration" 으로 죽었다. 직접 호출하고 실패 시
        # 예외를 전파해 상위(evaluate.py)에서 명확한 에러로 처리하도록 한다.
        hf_down(model_name, path)
    except Exception as e:
        print(e)
        if os.path.exists(path):
            shutil.rmtree(path)
        raise
