"""下载并准备多能谱仿真所需的 BrainWeb 1 mm 标签模体。

脚本不依赖本机路径：数据写入本脚本所在示例的 ``inputs/brainweb`` 目录。
默认下载 crisp 标签体，并可额外生成中心 33 层的小型回归测试输入。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inputs" / "brainweb" / "source"
TARGET = ROOT / "inputs" / "brainweb"
URL = "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"
ALIAS = "phantom_1.0mm_normal_crisp"
ARCHIVE = f"{ALIAS}.rawb.gz"
FULL_RAW = "brainweb_normal_1mm_181x217x181_uint8.raw"
CENTER_RAW = "brainweb-normal-1mm-center33-181x217x33-u8.raw"
FULL_SIZE = 181 * 217 * 181
EXPECTED_GZIP_SHA256 = "a3faa4b414d1943a6f4299e01a914442b3e72d2c89b049170c092610038d2210"
EXPECTED_RAW_SHA256 = "8693f9fde4f2b233237f2c5d0c4e4e2aac705bb08efd1e13ae70079ab7456c54"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(path: Path, force: bool) -> None:
    if path.exists() and not force:
        return
    form = urlencode({
        "do_download_alias": ALIAS, "format_value": "raw_byte",
        "zip_value": "gnuzip", "download_for_real": "[Start download!]",
    }).encode("ascii")
    request = Request(URL, data=form, headers={"User-Agent": "ykcbct-brainweb-downloader/1.0"}, method="POST")
    part = path.with_suffix(path.suffix + ".part")
    part.unlink(missing_ok=True)
    print(f"Downloading BrainWeb: {ALIAS}")
    with urlopen(request, timeout=300) as response, part.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    part.replace(path)


def prepare_full(archive: Path, output: Path, force: bool) -> None:
    if digest(archive) != EXPECTED_GZIP_SHA256:
        raise RuntimeError("BrainWeb archive SHA256 mismatch")
    if not output.exists() or force:
        part = output.with_suffix(output.suffix + ".part")
        with gzip.open(archive, "rb") as src, part.open("wb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(block)
        part.replace(output)
    if output.stat().st_size != FULL_SIZE or digest(output) != EXPECTED_RAW_SHA256:
        raise RuntimeError("BrainWeb label volume size or SHA256 mismatch")


def prepare_center33(full: Path, output: Path, force: bool) -> None:
    if output.exists() and not force:
        return
    # 原始布局为 [z][y][x]，取中心 z=74..106 共 33 层。
    row = 181 * 217
    first = 74 * row
    count = 33 * row
    with full.open("rb") as src:
        src.seek(first)
        data = src.read(count)
    if len(data) != count:
        raise RuntimeError("BrainWeb volume is too short for center-33 extraction")
    output.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BrainWeb labels for multispectrum_sim")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    parser.add_argument("--no-center33", action="store_true", help="skip the center-33 test volume")
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    TARGET.mkdir(parents=True, exist_ok=True)
    archive = SOURCE / ARCHIVE
    full = TARGET / FULL_RAW
    download(archive, args.force)
    prepare_full(archive, full, args.force)
    if not args.no_center33:
        prepare_center33(full, TARGET / CENTER_RAW, args.force)
    print(f"BrainWeb data ready: {TARGET}")


if __name__ == "__main__":
    main()
