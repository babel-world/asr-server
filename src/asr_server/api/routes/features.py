import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse

from asr_server.api.deps import valid_npy_file
from asr_server.schemas.features import FeaturesModelStateResponseBody
from asr_server.service.features.chinese_hubert_base import (
    extract_upload,
    features_start,
    features_stop,
)
from asr_server.service.features.speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common import (
    extract_upload as sv_extract_upload,
    features_start as sv_features_start,
    features_stop as sv_features_stop,
)
from asr_server.infra.worker.errors import WorkerSpawnFailed, WorkerSpawnTimeout

router = APIRouter(prefix="/features", tags=["features"])


def _cleanup_features_session(session_dir: Path) -> None:
    shutil.rmtree(session_dir, ignore_errors=True)


@router.post(
    "/chinese-hubert-base",
    response_class=FileResponse,
    summary="从 waveform .npy 提取 chinese-hubert-base 特征，返回 feature .npy",
    description=(
        "上传 GPT-SoVITS 约定预处理后的 mono float32 16 kHz 波形 ``.npy``，"
        "形状 ``(T,)`` 或 ``(1, T)``。"
        "响应为 ``last_hidden_state`` 特征 ``.npy``，形状 ``(1, T', 768)`` float32（BTC layout）。"
        "首次调用会自动启动长驻 worker（无需先调 ``/start``）；批量处理结束后请调用 ``/stop`` 释放显存。"
    ),
)
async def extract_chinese_hubert_base_features(
    background_tasks: BackgroundTasks,
    file=Depends(valid_npy_file),
) -> FileResponse:
    try:
        output_path, session_dir, download_name = await extract_upload(file)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except WorkerSpawnTimeout as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except WorkerSpawnFailed as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=e.stderr_tail or str(e),
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    background_tasks.add_task(_cleanup_features_session, session_dir)

    return FileResponse(
        path=output_path,
        media_type="application/octet-stream",
        filename=download_name,
    )


@router.post(
    "/chinese-hubert-base/start",
    response_model=FeaturesModelStateResponseBody,
    summary="预加载 chinese-hubert-base worker（可选）",
    description=(
        "显式启动长驻 worker 并加载模型。"
        "若不调用，本接口会在首次 ``POST /chinese-hubert-base`` 时自动启动。"
    ),
)
async def chinese_hubert_base_start() -> FeaturesModelStateResponseBody:
    try:
        return await features_start()
    except WorkerSpawnTimeout as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except WorkerSpawnFailed as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=e.stderr_tail or str(e),
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/chinese-hubert-base/stop",
    response_model=FeaturesModelStateResponseBody,
    summary="释放 chinese-hubert-base worker",
    description="结束长驻 worker 子进程并释放 GPU/内存。批量处理完成后调用一次即可。",
)
async def chinese_hubert_base_stop() -> FeaturesModelStateResponseBody:
    return await features_stop()


@router.post(
    "/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common",
    response_class=FileResponse,
    summary="从 sv_input .npy 提取 ERes2NetV2 SV embedding，返回 sv_output .npy",
    description=(
        "上传 GPT-SoVITS 管道 B 约定预处理后的 mono float32 16 kHz 波形 ``.npy``，"
        "形状 ``(T,)`` 或 ``(1, T)``，振幅为常规 float32（**不要**使用 1145.14 标度）。"
        "响应为 SV embedding ``.npy``，形状 ``(1, 20480)`` float32。"
        "首次调用会自动启动长驻 worker（无需先调 ``/start``）；批量处理结束后请调用 ``/stop`` 释放显存。"
    ),
)
async def extract_speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common_features(
    background_tasks: BackgroundTasks,
    file=Depends(valid_npy_file),
) -> FileResponse:
    try:
        output_path, session_dir, download_name = await sv_extract_upload(file)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except WorkerSpawnTimeout as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except WorkerSpawnFailed as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=e.stderr_tail or str(e),
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    background_tasks.add_task(_cleanup_features_session, session_dir)

    return FileResponse(
        path=output_path,
        media_type="application/octet-stream",
        filename=download_name,
    )


@router.post(
    "/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common/start",
    response_model=FeaturesModelStateResponseBody,
    summary="预加载 speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common worker（可选）",
    description=(
        "显式启动长驻 worker 并加载模型。"
        "若不调用，本接口会在首次 ``POST /speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common`` 时自动启动。"
    ),
)
async def speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common_start() -> FeaturesModelStateResponseBody:
    try:
        return await sv_features_start()
    except WorkerSpawnTimeout as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except WorkerSpawnFailed as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=e.stderr_tail or str(e),
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common/stop",
    response_model=FeaturesModelStateResponseBody,
    summary="释放 speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common worker",
    description="结束长驻 worker 子进程并释放 GPU/内存。批量处理完成后调用一次即可。",
)
async def speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common_stop() -> FeaturesModelStateResponseBody:
    return await sv_features_stop()
