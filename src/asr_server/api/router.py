from fastapi import APIRouter

from asr_server.api.routes import slice, transcribe

api_router = APIRouter(prefix="/api")
api_router.include_router(transcribe.router)
api_router.include_router(slice.router)
