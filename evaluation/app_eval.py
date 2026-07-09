import os
import json
import uvicorn
from multiprocessing import Process, Queue

from fastapi import FastAPI, Request
from fastapi.responses import Response
from evaluate import eval

app = FastAPI()


@app.post("/evaluate")
async def request(request: Request) -> Response:
    input = await request.json()
    task_id = input["task_id"]
    args = input["args"]

    queue = Queue()
    process = Process(target=eval, args=(task_id, args, queue))
    process.start()
    process.join()

    result = queue.get()

    return Response(
        content=json.dumps(dict(result), indent=4), media_type="application/json"
    )


if __name__ == "__main__":
    # 서버 실행
    api_host = "0.0.0.0"
    api_port = int(os.getenv("EVAL_PORT", "11189"))
    uvicorn.run(app, host=api_host, port=api_port)
