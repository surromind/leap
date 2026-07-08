import os
import gradio as gr

# --- Fix: gradio_client 1.4.2 가 boolean JSON schema 값을 처리하지 못하는 문제 패치 ---
# 최신 pydantic/fastapi 는 dict/Any 필드에 대해 `additionalProperties: true` 같은
# boolean 스키마를 생성한다. gradio_client 의 스키마 파서가 이를 dict 로 가정하고
# `"const" in schema` 를 수행하다 `TypeError: argument of type 'bool' is not iterable`
# 를 일으키고, 그 결과 API 정보 생성(/ 라우트)이 500 을 반환 -> Gradio launch() 의
# localhost self-check 실패 -> "When localhost is not accessible ..." ValueError 로
# 프론트엔드 컨테이너가 재시작 루프에 빠진다. 아래 패치로 boolean 스키마를 안전하게 처리한다.
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "bool" if schema else "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
# --- end fix ---

from css import css
from components import (
    tuning_tab,
    evaluate_tab,
    leaderboard_tab,
    tuning_dataset_tab,
    tuning_history_tab,
    evaluate_dataset_tab,
    gpu_tab,
)
from event_listener import status_listener, queue_listener
from config import (
    get_eval_datasets,
    get_tuning_datasets,
    get_eval_models,
    get_tuning_models,
)


def build_demo():
    with gr.Blocks(title="SURROMIND", css=css) as demo:
        gr.Markdown("# LEAP <br><br>")
        EVAL_DATASETS = get_eval_datasets()
        TUNING_DATASETS = get_tuning_datasets()
        EVAL_MODELS = get_eval_models()
        TUNING_MODELS = get_tuning_models()
        trigger = gr.Textbox(
            value=status_listener.trigger_change, visible=False, every=0.2
        )
        leaderboard_tab.create_leaderboard_tab()
        evaluate = evaluate_tab.create_evaluate_tab(EVAL_MODELS, EVAL_DATASETS)
        evaluate_dataset_tab.create_evaluate_dataset_tab(EVAL_DATASETS)
        tuning = tuning_tab.create_tuning_tab(TUNING_MODELS, TUNING_DATASETS)
        tuning_dataset_tab.create_tuning_dataset_tab(TUNING_DATASETS)
        tuning_history = tuning_history_tab.create_tuning_history_tab()
        gpu_tab.create_gpu_tab()

        # Queue Management
        trigger.change(
            fn=status_listener.update_display,
            inputs=[],
            outputs=[
                evaluate["current_task"],
                evaluate["queue"],
                evaluate["running_args"],
                evaluate["output_box"],
                evaluate["progress_bar"],
                tuning["current_task"],
                tuning["queue"],
                tuning["running_args"],
                tuning["output_box"],
                tuning["progress_bar"],
                tuning["loss_graph"],
                tuning["grad_norm_graph"],
                evaluate["gpu_status"],
                tuning["gpu_status"],
            ],
            show_progress=False,
        )
        # Tuning History
        tuning["tuning_tab"].select(
            fn=queue_listener.update_dropdown,
            inputs=[tuning["queue"]],
            outputs=[
                tuning_history["finished_queue_elems"],
                tuning["pending_queue_elems"],
            ],
            show_progress=False,
        )
        tuning_history["tuning_history_tab"].select(
            fn=queue_listener.update_dropdown,
            inputs=[tuning["queue"]],
            outputs=[
                tuning_history["finished_queue_elems"],
                tuning["pending_queue_elems"],
            ],
            show_progress=False,
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=64).launch(
        share=False,
        server_name="0.0.0.0",
        debug=True,
        server_port=int(os.getenv("FRONTEND_PORT", 11184)),
    )
