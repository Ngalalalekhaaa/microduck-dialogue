"""Install a pinned local Qwen VLM without modifying Python or system packages.

Only official Qwen, llama.cpp and Ubuntu artifacts are downloaded. The bundled
Ubuntu loader allows the official llama.cpp binary to run on Ubuntu 22.04.
CUDA libraries, when used, are read from the existing microduck environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / ".cache" / "microduck_vlm"
REVISION = "52d6c8ffea26cc873ac5ad116f8631268d7eb503"
LLAMA_TAG = "b12375"
LLAMA_COMMIT = "9d25e0014c4b4b316873c01f4938c0e3c687287d"
MAX_CACHE_BYTES = 3_000_000_000
MIN_FREE_BYTES = 1_500_000_000
MODEL = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ = "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
ARCHIVE = f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
HF = f"https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/{REVISION}"
ARTIFACTS = [
    (
        ARCHIVE,
        199478619,
        "36ad425cb846ec831ec48b7789b032e43e6df931f95eed7fcbd7fabdbd521225",
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{ARCHIVE}",
    ),
    (
        MODEL,
        1107409952,
        "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae",
        f"{HF}/{MODEL}?download=true",
    ),
    (
        MMPROJ,
        445053216,
        "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82",
        f"{HF}/{MMPROJ}?download=true",
    ),
    (
        "libc6_2.39-0ubuntu8.12_amd64.deb",
        3264258,
        "d821b41fc59cb4978ecd53bc83e5c964ac15d305c8d2ac4c8a35aebe55010fb9",
        "https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8.12_amd64.deb",
    ),
    (
        "libgcc-s1_14.2.0-4ubuntu2~24.04.1_amd64.deb",
        78392,
        "aa7fadbe33b78bcf99885318040601c550c208929565b179891d9a3cc2aa68cd",
        "https://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libgcc-s1_14.2.0-4ubuntu2~24.04.1_amd64.deb",
    ),
    (
        "libstdc++6_14.2.0-4ubuntu2~24.04.1_amd64.deb",
        792064,
        "a51f8de7829211db961a31f02158058ad1a95f92ac6d0a5dff6350e2821c54c0",
        "https://archive.ubuntu.com/ubuntu/pool/main/g/gcc-14/libstdc++6_14.2.0-4ubuntu2~24.04.1_amd64.deb",
    ),
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def cache_bytes() -> int:
    return sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file() and not p.is_symlink())


def check_space(extra: int = 0) -> None:
    used = cache_bytes()
    free = shutil.disk_usage(CACHE).free
    if used + extra > MAX_CACHE_BYTES or free - extra < MIN_FREE_BYTES:
        raise RuntimeError(
            f"下载已停止：缓存预算 3 GB，磁盘需保留 1.5 GB；当前缓存 {used} 字节，"
            f"空闲 {free} 字节，还需 {extra} 字节。请提供更大磁盘后再部署。"
        )


def download(artifact: tuple[str, int, str, str]) -> None:
    name, size, sha, url = artifact
    path = CACHE / name
    if path.exists():
        if path.stat().st_size != size or digest(path) != sha:
            raise RuntimeError(f"缓存校验失败：{path}；请先检查该文件，程序不会覆盖它。")
        print(f"Verified {name}", flush=True)
        return
    partial = path.with_name(path.name + ".part")
    total = partial.stat().st_size if partial.exists() else 0
    if total > size:
        raise RuntimeError(f"临时文件超过固定大小：{partial}；请检查文件后重试。")
    check_space(size - total)
    last_report = time.monotonic()
    request_bytes = 32 * 1024 * 1024
    attempts = 5 + (size + request_bytes - 1) // request_bytes
    for attempt in range(attempts):
        if total == size:
            break
        separator = "&" if "?" in url else "?"
        request = urllib.request.Request(
            f"{url}{separator}offset={total}",
            headers={"Range": f"bytes={total}-{min(total + request_bytes, size) - 1}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("ab") as stream:
                if total and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith(f"bytes {total}-")
                ):
                    raise RuntimeError(f"服务器未按要求续传 {name}；临时文件已保留。")
                while block := response.read(4 * 1024 * 1024):
                    if total + len(block) > size:
                        raise RuntimeError(f"下载超过固定大小：{name}")
                    check_space(len(block))
                    stream.write(block)
                    total += len(block)
                    if time.monotonic() - last_report > 20:
                        print(f"Downloading {name}: {total}/{size}", flush=True)
                        last_report = time.monotonic()
        except OSError:
            if attempt == attempts - 1:
                raise
            total = partial.stat().st_size if partial.exists() else 0
    if total != size or digest(partial) != sha:
        raise RuntimeError(f"下载校验失败：{partial}；请检查网络和临时文件后重试。")
    partial.rename(path)
    print(f"Verified {name}", flush=True)


def extract_runtime() -> None:
    """Extract trusted archives inside CACHE; never run a package install hook."""
    with tarfile.open(CACHE / ARCHIVE) as archive:
        members = [
            item
            for item in archive
            if len(PurePosixPath(item.name).parts) == 2 and (".so" in item.name or item.name.endswith("/llama-server"))
        ]
        missing = sum(item.size for item in members if not (CACHE / item.name).exists())
        check_space(missing + 25_000_000)
        for item in members:
            if PurePosixPath(item.name).parts[0] != f"llama-{LLAMA_TAG}":
                raise RuntimeError("Unexpected archive directory")
            target = CACHE / item.name
            target.parent.mkdir(exist_ok=True)
            if item.isfile():
                if target.is_symlink():
                    raise RuntimeError(f"Unexpected symlink: {target}")
                with archive.extractfile(item) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                target.chmod(item.mode & 0o755)
            elif item.issym():
                if "/" in item.linkname or item.linkname in {".", ".."}:
                    raise RuntimeError("Unexpected archive symlink")
                if target.is_symlink() and os.readlink(target) == item.linkname:
                    continue
                target.symlink_to(item.linkname)
            else:
                raise RuntimeError("Unexpected archive member")
    for name, _, _, _ in ARTIFACTS:
        if name.endswith(".deb"):
            subprocess.run(["dpkg-deb", "-x", str(CACHE / name), str(CACHE / "runtime")], check=True)
    check_space()


def cuda_libraries() -> list[str]:
    site = Path.home() / "microduck/microduck_rl/.venv/lib/python3.12/site-packages/nvidia"
    directories = [site / "cublas/lib", site / "cuda_runtime/lib"]
    if all(p.is_dir() for p in directories):
        return [str(p) for p in directories]
    return []


def loader_command() -> list[str]:
    library = CACHE / "runtime/usr/lib/x86_64-linux-gnu"
    binary_dir = CACHE / f"llama-{LLAMA_TAG}"
    paths = [str(library), str(binary_dir), *cuda_libraries()]
    return [str(library / "ld-linux-x86-64.so.2"), "--library-path", ":".join(paths), str(binary_dir / "llama-server")]


def write_manifest() -> None:
    files = {}
    for directory in (CACHE / f"llama-{LLAMA_TAG}", CACHE / "runtime"):
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink():
                files[str(path.relative_to(CACHE))] = digest(path)
    payload = {
        "model": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "revision": REVISION,
        "llama_tag": LLAMA_TAG,
        "llama_commit": LLAMA_COMMIT,
        "license": "Apache-2.0",
        "artifacts": [{"name": n, "bytes": s, "sha256": h, "url": u} for n, s, h, u in ARTIFACTS],
        "runtime_files_sha256": files,
        "cuda_libraries_read_only": cuda_libraries(),
        "cache_bytes": cache_bytes(),
    }
    (CACHE / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_deployment(command: list[str], backend: str) -> None:
    artifacts = {name: {"sha256": sha, "bytes": size} for name, size, sha, _ in ARTIFACTS}
    payload = {
        "model_alias": "microduck-vlm",
        "model_repository": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "model_revision": REVISION,
        "model": {"file": MODEL, "quantization": "Q4_K_M", **artifacts[MODEL]},
        "vision_encoder": {"file": MMPROJ, "quantization": "Q8_0", **artifacts[MMPROJ]},
        "model_license": "Apache-2.0",
        "llama_cpp": {"tag": LLAMA_TAG, "commit": LLAMA_COMMIT, "archive": ARCHIVE, **artifacts[ARCHIVE]},
        "backend": backend,
        "working_directory": str(CACHE / f"llama-{LLAMA_TAG}"),
        "command": command,
        "command_shell": shlex.join(command),
    }
    (CACHE / "deployment.json").write_text(json.dumps(payload, indent=2) + "\n")


def verify() -> None:
    for name, size, expected, _ in ARTIFACTS:
        path = CACHE / name
        if not path.is_file() or path.stat().st_size != size or digest(path) != expected:
            raise RuntimeError(f"缺少或损坏的文件：{path}；请运行 setup_vlm.py --download。")
    manifest = json.loads((CACHE / "manifest.json").read_text())
    for name, expected in manifest["runtime_files_sha256"].items():
        path = (CACHE / name).resolve()
        if not path.is_relative_to(CACHE.resolve()) or not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"运行库校验失败：{name}；请重新运行 setup_vlm.py --download。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--download", action="store_true", help="Download, verify and extract pinned artifacts")
    modes.add_argument("--verify", action="store_true", help="Verify installed artifacts without a network request")
    modes.add_argument("--serve", action="store_true", help="Verify then serve on loopback; Ctrl+C stops it")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU offload")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 1024 <= args.context <= 8192 or not 1 <= args.threads <= 32:
        parser.error("port must be 1024..65535; context 1024..8192; threads 1..32")
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.download:
        if not shutil.which("dpkg-deb"):
            raise RuntimeError("需要 dpkg-deb 解包运行库；当前部署脚本适用于 Ubuntu/Debian x86_64。")
        missing = sum(
            size - ((CACHE / (name + ".part")).stat().st_size if (CACHE / (name + ".part")).exists() else 0)
            for name, size, _, _ in ARTIFACTS
            if not (CACHE / name).exists()
        )
        check_space(missing + (300_000_000 if not (CACHE / "manifest.json").exists() else 0))
        for artifact in ARTIFACTS:
            download(artifact)
        extract_runtime()
        subprocess.run(loader_command() + ["--version"], check=True)
        write_manifest()
        print(f"Installed and verified: {CACHE}")
        return
    verify()
    if args.verify:
        print(f"Verified local VLM: {CACHE}")
        return
    use_cuda = not args.cpu and bool(cuda_libraries())
    command = loader_command() + [
        "--model",
        str(CACHE / MODEL),
        "--mmproj",
        str(CACHE / MMPROJ),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--alias",
        "microduck-vlm",
        "--ctx-size",
        str(args.context),
        "--parallel",
        "1",
        "--threads",
        str(args.threads),
        "--n-gpu-layers",
        "99" if use_cuda else "0",
        "--jinja",
        "--no-webui",
        "--image-max-tokens",
        "512",
        "--cache-ram",
        "256",
    ]
    if not use_cuda:
        command.append("--no-mmproj-offload")
    write_deployment(command, "CUDA" if use_cuda else "CPU")
    print(f"Backend: {'CUDA' if use_cuda else 'CPU'}; Ctrl+C stops the server.", flush=True)
    print(shlex.join(command), flush=True)
    clean_env = {k: v for k, v in os.environ.items() if k not in {"LD_PRELOAD", "LD_LIBRARY_PATH"}}
    # The isolated loader becomes /proc/self/exe, so backend discovery uses cwd.
    os.chdir(CACHE / f"llama-{LLAMA_TAG}")
    os.execve(command[0], command, clean_env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
