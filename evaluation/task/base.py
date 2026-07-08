import os
from abc import ABC, abstractmethod

import datasets

from .utils import positional_deprecated


class Task(ABC):
    """
    데이터셋, 문제, 정답, 평가 방법을 포함한 추상 클래스
    """

    # 객체 변수, 모두 정의할 필요 없음
    VERSION: int = 0
    IN_HF_HUB: bool = False  # 허깅페이스 허브에 업로드되어 있는 데이터셋인지 여부
    DATASET_PATH: str = None  # 데이터셋 폴더 경로  (e.g., "bm_kobest_v1", "bm_klue")
    DATASET_NAME: str = None  # 데이터셋 이름  (e.g., "boolq", "nli")
    DATA_FILES: dict = (
        None  # 데이터셋 파일 이름 (e.g., {"train": "train.jsonl", "validation": "validation.jsonl"})
    )
    DATASET_TYPE: str = None  # 파일 타입 (e.g., "json", "csv")
    DOWNLOAD_OPTIONS: dict = (
        None  # 데이터셋 다운로드 옵션 (e.g., {"delimiter": "\t", "quoting": 3})
    )
    # seaweedfs 로 제공되던 커스텀 데이터셋 폴더(bm_*)가 없을 때(또는 해당 config 가
    # 로컬에 없을 때) 대신 로드할 공개 HF 허브 데이터셋 매핑. DATASET_PATH -> 허브 repo_id.
    HF_HUB_FALLBACK: dict = {
        "bm_kobest_v1": "skt/kobest_v1",
        "bm_klue": "klue/klue",
        "bm_ko-MMLU": "HAERAE-HUB/KMMLU",
    }

    def __init__(self, data_dir: str = None, cache_dir=None):
        """
        Args:
            data_dir: 데이터 파일이 포함된 로컬 폴더 경로 (default: None)
            cache_dir: 캐시 폴더 경로 (default: None)
                HuggingFace `datasets` API를 따르며 기본 캐시 디렉토리는 다음 경로에 위치합니다: `~/.cache/huggingface/datasets`
        """
        self.load(data_dir, cache_dir)  # 데이터 다운로드
        self._training_docs = None
        self._fewshot_docs = None

    def load(self, data_dir=None, cache_dir=None):
        """
        데이터셋 다운로드
        """
        # huggingface hub에 있는 데이터셋을 로드하는 경우
        if self.IN_HF_HUB:
            local_path = os.path.join(data_dir, self.DATASET_PATH)
            try:
                # 로컬에 재구성된 커스텀 데이터셋 폴더가 있으면 우선 사용 (오프라인/기존 방식)
                self.dataset = datasets.load_dataset(
                    local_path,
                    self.DATASET_NAME,
                    cache_dir=cache_dir,
                )  # 로컬 폴더명만 지정해도 load_dataset 가능
            except Exception as local_err:
                # seaweedfs 커스텀 데이터셋이 없거나 해당 config 가 로컬에 없으면 공개 HF 허브로 fallback
                fallback_repo = self.HF_HUB_FALLBACK.get(self.DATASET_PATH)
                if not fallback_repo:
                    raise local_err
                self.dataset = datasets.load_dataset(
                    fallback_repo,
                    self.DATASET_NAME,
                    cache_dir=cache_dir,
                )

        # 로컬 데이터셋을 로드하는 경우
        else:
            base_path = os.path.join(data_dir, self.DATASET_PATH, self.DATASET_NAME)
            data_files = {k: base_path + "/" + v for k, v in self.DATA_FILES.items()}
            self.dataset = datasets.load_dataset(
                self.DATASET_TYPE, data_files=data_files, **self.DOWNLOAD_OPTIONS
            )  # json 파일 경로 직접 지정
    

    def n_shot(self):
        """데이터셋마다 적합한 fewshot 수 반환"""
        return 0

    def request_type(self):
        """
        데이터셋 유형 반환

        - loglikelihood: 모델이 계산한 선택지들의 로그 확률을 정답 선택지와 비교해 평가
        - greedy_until: 모델이 생성한 문장을 정답 문장과 비교해 평가
        """
        return "loglikelihood"

    def description(self):
        """fewshot 예제에 앞서 추가될 작업(Task)의 설명"""
        return None

    @abstractmethod
    def has_training_docs(self):
        """학습 세트를 가지고 있는지 여부"""
        pass

    @abstractmethod
    def has_validation_docs(self):
        """검증 세트를 가지고 있는지 여부"""
        pass

    @abstractmethod
    def has_test_docs(self):
        """테스트 세트를 가지고 있는지 여부"""
        pass

    def training_docs(self):
        """
        Returns:
            Iterable[obj]: doc_to_text에서 처리할 수 있는 모든 객체의 iterable
        """
        return []

    def validation_docs(self):
        """
        Returns:
            Iterable[obj]: doc_to_text에서 처리할 수 있는 모든 객체의 iterable
        """
        return []

    def test_docs(self):
        """
        Returns:
            Iterable[obj]: doc_to_text에서 처리할 수 있는 모든 객체의 iterable
        """
        return []

    def _process_doc(self, doc):
        """
        개별 데이터 샘플을 처리하는 함수 (detokenize, strip, replace 등)
        분할된 데이터에 대해 map 연산으로 적용 가능
        E.g. `map(self._process_doc, self.dataset["validation"])`

        Returns:
            Dict: 처리된 데이터 샘플
        """
        return doc

    def fewshot_examples(self, k, rnd):
        """
        k개의 랜덤 샘플을 반환
        """
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    @abstractmethod
    def doc_to_text(self, doc):
        """
        Dict 형태의 데이터 샘플을 모델에 입력하기 위해 텍스트로 변환하는 함수
        """
        pass

    @abstractmethod
    def doc_to_target(self, doc):
        """
        Dict 형태의 데이터 샘플에서 정답 문장 및 선택지를 추출하는 함수
        """
        pass

    @abstractmethod
    def construct_requests(self, doc, ctx):
        """
        RequestFactory를 사용해서 Request 객체를 생성하고, 모델에 입력될 Request 객체의 Iterable를 반환
        Args:
            doc: training_docs, validation_docs 또는 test_docs에서 반환된 데이터 샘플
            ctx: fewshot_context에 의해 생성된 컨텍스트 문자열
                (몇 가지 샘플 예시, doc의 질문 부분 포함)
        """
        pass

    @abstractmethod
    def process_results(self, doc, results):
        """
        단일 데이터 샘플과 모델의 예측 결과를 받아 평가 지표를 계산하는 함수
        Args:
            doc: training_docs, validation_docs 또는 test_docs에서 반환된 데이터 샘플
            results: 모델의 예측 결과
        Returns:
            Dict[str, Any]: 평가 지표 이름을 key로 하고, 평가 지표 값을 value로 하는 딕셔너리
        """
        pass

    @abstractmethod
    def aggregation(self):
        """
        Returns:
            Dict[str, Callable]: 평가 지표 이름을 key로 하고, 평가 지표 값 리스트를 집계하는 함수를 value로 하는 딕셔너리
        """
        pass

    @abstractmethod
    def higher_is_better(self):
        """
        메트릭에 대해 값이 높을수록 좋은지 여부를 정의
        Returns:
            Dict[str, bool]: 평가 지표 이름을 key로 하고, 높은 값이 더 좋은 평가 지표인지 여부를 value로 하는 딕셔너리
        """
        pass

    @positional_deprecated
    def fewshot_context(self, doc, num_fewshot, rnd, description=None):
        """
        제공된 설명(있는 경우), `num_fewshot` 개수의 예제, 그리고 추가된 프롬프트 예제로 구성된
        fewshot 컨텍스트 문자열을 반환

        Args:
            doc (str): training_docs, validation_docs, 또는 test_docs에서 반환된 문서
            num_fewshot (int): 반환된 컨텍스트 문자열에 포함할 fewshot 예제의 개수
            rnd (random.Random): 예제를 무작위로 샘플링하기 위해 사용되는 의사난수 생성기
            description (str): fewshot 예제에 앞서 추가될 작업(Task)의 설명
        Returns:
            str: 생성된 fewshot 컨텍스트 문자열
        """
        assert rnd is not None
        description = description + "\n\n" if description else ""

        # num_fewshot가 0인 경우 예제가 없는 빈 문자열을 반환
        if num_fewshot == 0:
            labeled_examples = ""
        # num_fewshot가 0보다 큰 경우 예제를 무작위로 샘플링
        else:
            # 학습 세트가 있는 경우 학습 세트에서 예제를 샘플링
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            # 학습 세트가 없는 경우 검증 세트 또는 테스트 세트에서 예제를 샘플링
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )
                # doc과의 중복 방지를 위해 num_fewshot + 1개의 예제를 샘플링 후 중복 제거
                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            # 샘플링한 예제들을 텍스트로 변환
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        # doc의 질문 부분
        example = self.doc_to_text(doc)

        return description + labeled_examples + example


REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 2,  # loglikelihood 요청은 2개의 값을 반환합니다.
    "greedy_until": None,  # greedy_until 요청은 여러 값을 반환하지 않습니다.
}


class Request:
    """
    특정 유형의 요청(Request)을 나타내는 클래스
    요청 유형, 인수, 그리고 (선택적으로) 인덱스를 포함
    """

    def __init__(self, request_type, args, index=None):
        """
        Request 객체 초기화

        Args:
            request_type (str): 요청의 유형 (e.g., "loglikelihood", "greedy_until")
            args (tuple): 요청에 필요한 인수
            index(int): 요청의 인덱스 (기본값: None)
        """
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError(
                "The request type {} is not implemented!".format(request_type)
            )

        self.request_type = request_type
        self.args = args
        self.index = index

    def __iter__(self):
        """
        요청 객체를 반복 가능하도록 구현
        요청 유형이 다중 값을 반환하는 경우, 각 요청을 반복적으로 생성
        """
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        for i in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, i)

    def __getitem__(self, i):
        """
        특정 인덱스에 해당하는 요청 객체를 반환
        """
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        return Request(self.request_type, self.args, i)

    def __eq__(self, other):
        """
        두 요청 객체를 비교하여 동일한지 여부를 반환
        """
        return (
            self.request_type == other.request_type
            and self.args == other.args
            and self.index == other.index
        )

    def __repr__(self):
        """
        요청 객체를 문자열로 표현
        """
        return f"Req_{self.request_type}{self.args}[{self.index}]\n"


class RequestFactory:
    """
    Request 객체를 동적으로 생성하는 팩토리 클래스
    요청 유형을 속성처럼 호출하여 Request 객체를 생성
    """

    def __getattr__(self, attr):
        """
        호출된 속성 이름을 사용하여 Request 객체를 생성하는 함수

        Args:
            attr(str): 호출된 속성 이름 (e.g., "loglikelihood", "greedy_until")
        Returns:
            callable: Request 객체를 반환하는 함수
        """

        def fn(*args):
            return Request(attr, args)

        return fn


# RequestFactory 인스턴스 생성
rf = RequestFactory()
