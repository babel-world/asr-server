from fastapi import FastAPI

from asr_server.api.router import api_router

app = FastAPI()
app.include_router(api_router)


@app.get("/")
def hello() -> dict[str, str]:
    return {"message": "Hello World"}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "asr_server.main:app", host="127.0.0.1", port=19031, reload=True
    )
