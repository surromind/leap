import os

API_URL = f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('BACKEND_PORT', 11188)}"
TYPE_URL = {
    "evaluate": f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('EVAL_PORT', 11189)}/evaluate",
    "tuning": f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('TUNING_PORT', 11190)}/tuning",
}
