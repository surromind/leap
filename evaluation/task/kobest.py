import numpy as np
from .base import rf, Task
from .metrics import macro_f1_score, mean


class BoolQ(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_kobest_v1"
    DATASET_NAME = "boolq"

    def n_shot(self):
        return 0

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "내용: {}\n질문: 다음 문장이 내용에 있는 사실과 일치하는가?\n문장: {}\n응답:".format(
            doc["paragraph"], doc["question"]
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오", 1: "예"}[doc["label"]])

    def idx_to_target(self, idx):
        return {0: "아니오", 1: "예"}[idx]

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오")
        ll_yes, _ = rf.loglikelihood(ctx, " 예")

        return ll_no, ll_yes

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {"acc": pred == gold, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "macro_f1": True}

    def aggregation(self):
        return {"acc": mean, "macro_f1": macro_f1_score}


class COPA(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_kobest_v1"
    DATASET_NAME = "copa"

    def n_shot(self):
        return 3

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        """
        Connector: “왜냐하면” if Question is “원인” else “그래서”
        Format: “{Premise} {Connector} {Answer Alternative}”
        """
        connector = {
            "원인": "왜냐하면",
            "결과": "그래서",
        }[doc["question"].strip()]

        return doc["premise"] + f" {connector}"

    def doc_to_target(self, doc):
        correct_choice = (
            doc["alternative_1"] if doc["label"] == 0 else doc["alternative_2"]
        )

        return " " + correct_choice

    def idx_to_target(self, idx):
        return {0: "alternative_1", 1: "alternative_2"}[idx]

    def construct_requests(self, doc, ctx):
        ll_choice1, _ = rf.loglikelihood(ctx.strip(), " " + doc["alternative_1"])
        ll_choice2, _ = rf.loglikelihood(ctx.strip(), " " + doc["alternative_2"])

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {"acc": pred == gold, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "macro_f1": True}

    def aggregation(self):
        return {"acc": mean, "macro_f1": macro_f1_score}


class WiC(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_kobest_v1"
    DATASET_NAME = "wic"

    def n_shot(self):
        return 3

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "문장1: {}\n문장2: {}\n질문: 문장1과 문장2에서 쓰인 단어 [{}]가 같은 뜻으로 쓰였나?\n응답:".format(
            doc["context_1"], doc["context_2"], doc["word"]
        )

    def doc_to_target(self, doc):
        return "{}".format({0: " 아니오", 1: " 예"}[doc["label"]])

    def idx_to_target(self, idx):
        return {0: " 아니오", 1: " 예"}[idx]

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오")
        ll_yes, _ = rf.loglikelihood(ctx, " 예")

        return ll_no, ll_yes

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {"acc": pred == gold, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "macro_f1": True}

    def aggregation(self):
        return {"acc": mean, "macro_f1": macro_f1_score}


class HellaSwag(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_kobest_v1"
    DATASET_NAME = "hellaswag"

    def n_shot(self):
        return 0

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "{}".format(doc["context"]),
            "choices": [
                doc["ending_1"],
                doc["ending_2"],
                doc["ending_3"],
                doc["ending_4"],
            ],
            "gold": int(doc["label"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]

        acc = 1.0 if pred == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {"acc": acc, "acc_norm": acc_norm, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "acc_norm": True, "macro_f1": True}

    def aggregation(self):
        return {"acc": mean, "acc_norm": mean, "macro_f1": macro_f1_score}


class SentiNeg(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_kobest_v1"
    DATASET_NAME = "sentineg"

    def n_shot(self):
        return 5

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}".format(doc["sentence"])

    def doc_to_target(self, doc):
        return " ({})".format({0: "부정", 1: "긍정"}[doc["label"]])

    def idx_to_target(self, idx):
        return {0: "부정", 1: "긍정"}[idx]

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " (부정)")
        ll_yes, _ = rf.loglikelihood(ctx, " (긍정)")

        return ll_no, ll_yes

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {"acc": pred == gold, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "macro_f1": True}

    def aggregation(self):
        return {"acc": mean, "macro_f1": macro_f1_score}
