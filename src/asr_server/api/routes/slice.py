import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from asr_server.service.slice import slice_upload_to_zip

router = APIRouter(prefix="/slice", tags=["slice"])


def _cleanup_slice_session(session_dir: Path) -> None:
    shutil.rmtree(session_dir, ignore_errors=True)


@router.post(
    "",
    response_class=FileResponse,
    summary="对上传音频按静音边界切片，返回 ZIP 压缩包",
)
async def slice_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ..., description="音频文件（wav、mp3、flac 等 soundfile 支持的格式）"
    ),
    threshold_db: float = Form(
        default=-40.0, description="静音阈值（dB），越小越严格"
    ),
    min_length_ms: int = Form(
        default=5000, description="切片最短有效语音长度（毫秒）"
    ),
    min_interval_ms: int = Form(
        default=300, description="构成切分点的最短连续静音长度（毫秒）"
    ),
    hop_size_ms: int = Form(default=20, description="分帧步长（毫秒），决定时间分辨率"),
    max_sil_kept_ms: int = Form(
        default=5000, description="切分点两侧保留的最大静音余量（毫秒）"
    ),
) -> FileResponse:
    try:
        zip_path, session_dir, download_name = await slice_upload_to_zip(
            file,
            threshold_db=threshold_db,
            min_length_ms=min_length_ms,
            min_interval_ms=min_interval_ms,
            hop_size_ms=hop_size_ms,
            max_sil_kept_ms=max_sil_kept_ms,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法解析或处理音频: {e}",
        ) from e

    background_tasks.add_task(_cleanup_slice_session, session_dir)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=download_name,
    )
