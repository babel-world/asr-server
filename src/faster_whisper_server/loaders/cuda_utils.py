import os
import site
from pathlib import Path

_CUDA_DLLS_READY = False


def _setup_cuda_dlls() -> list[str]:
    """Register NVIDIA CUDA 12 DLL directories for CTranslate2 GPU inference.

    - Windows only: skipped on Linux/macOS (shared libs are found via LD_LIBRARY_PATH there).
    - Idempotent: subsequent calls return immediately without touching PATH again.
    - Uses site.getsitepackages() so it works regardless of venv layout (uv, venv, conda ...).
    - Both os.add_dll_directory (Python-layer) and PATH (CTranslate2 C++ engine) are updated,
      because ctranslate2 resolves cublas/cudnn at inference time through the Win32 search order.

    Returns:
        List of DLL directories that were registered.
    """
    global _CUDA_DLLS_READY
    if _CUDA_DLLS_READY:
        return []
    if os.name != "nt":
        _CUDA_DLLS_READY = True
        return []

    # nvidia-* packages install DLLs under <site-packages>/nvidia/<lib>/bin/
    # Cover the libs that ctranslate2 links against at runtime.
    _NVIDIA_SUBDIRS = (
        "cublas/bin",
        "cudnn/bin",
        "cuda_runtime/bin",
        "cuda_nvrtc/bin",
        "nvjitlink/bin",   # dependency of cudnn >=9
        "cusparse/bin",    # occasionally needed by newer ctranslate2
    )

    registered: list[str] = []
    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for sub in _NVIDIA_SUBDIRS:
            p = nvidia_dir / sub.replace("/", os.sep)
            if not p.is_dir():
                continue
            path_str = str(p)
            registered.append(path_str)
            try:
                os.add_dll_directory(path_str)
            except OSError:
                pass  # directory exists but add_dll_directory failed; PATH injection below still helps

    if registered:
        current_path = os.environ.get("PATH", "")
        # Prepend so our CUDA 12 libs take priority over any system-wide CUDA installation.
        os.environ["PATH"] = os.pathsep.join(registered) + os.pathsep + current_path

    _CUDA_DLLS_READY = True
    return registered


_setup_cuda_dlls()
