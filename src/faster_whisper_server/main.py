from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello() -> dict[str, str]:
    return {"message": "Hello World"}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "faster_whisper_server.main:app", host="127.0.0.1", port=8000, reload=True
    )
