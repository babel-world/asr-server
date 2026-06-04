from fastapi import APIRouter

from asr_server.api.routes import transcribe

api_router = APIRouter(prefix="/api")
api_router.include_router(transcribe.router)
