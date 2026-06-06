import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from asr_server.service.manifest import build_manifest_upload

router = APIRouter(prefix="/manifest", tags=["manifest"])


def _cleanup_manifest_session(session_dir: Path) -> None:
    shutil.rmtree(session_dir, ignore_errors=True)


@router.post(
    "",
    response_class=FileResponse,
    summary="从切片 ZIP 批量转录并生成 manifest CSV",
    description=(
        "请求体为 ``/api/audio/slice`` 返回的 ZIP（内含切片 WAV）。"
        "包内每个 WAV 须符合 ``{base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav``，"
        "且所有文件的 ``base_name`` 相同。"
        "响应为 ``{base_name}_manifest.csv``，表头: "
        "``filename,speaker,language,text,probability``。"
    ),
)
async def build_manifest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ..., description="切片 ZIP（通常为 {stem}_slices.zip）"
    ),
) -> FileResponse:
    try:
        csv_path, session_dir, download_name = await build_manifest_upload(file)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法解析或处理 ZIP: {e}",
        ) from e

    background_tasks.add_task(_cleanup_manifest_session, session_dir)

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=download_name,
    )
