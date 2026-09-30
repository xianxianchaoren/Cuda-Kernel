import importlib.util
import re
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = ROOT / ".build"


def module_name(name):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return f"attn_{safe}"


def discover_variants(root=ROOT):
    root = Path(root)
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "variant.py").is_file()
    )


def _load_manifest(vdir):
    spec = importlib.util.spec_from_file_location(f"variant_{vdir.name}", vdir / "variant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_variant(name, root=ROOT):
    vdir = Path(root) / name
    if not (vdir / "variant.py").is_file():
        raise FileNotFoundError(f"no variant.py found in {vdir}")

    manifest = _load_manifest(vdir)
    sources = [str(vdir / s) for s in manifest.SOURCES]
    build_dir = BUILD_ROOT / name
    build_dir.mkdir(parents=True, exist_ok=True)

    module = load(
        name=module_name(name),
        sources=sources,
        extra_cuda_cflags=list(getattr(manifest, "CUDA_FLAGS", ["-O2"])),
        build_directory=str(build_dir),
        verbose=False,
    )
    return module, getattr(manifest, "ENTRY", "forward")
