"""Fetch three preselected NYCU synchronized laboratory flights from the authors' public ZIP."""

import argparse
import hashlib
from pathlib import Path
import re
from urllib.request import Request, urlopen
import zipfile

from fetch_tracknetv3 import ArchiveWindow


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nycu-stereo6"
BASE = "https://drive.usercontent.google.com/download?id=1qQ25VeI4elPIpjuKtFM-oInnyuKbo8Mf&export=download&confirm=t"
TRIALS = ("001", "002", "003")
NAMES = ("CameraReaderL.avi", "CameraReaderL.cfg", "CameraReaderL_ball.csv",
         "CameraReaderR.avi", "CameraReaderR.cfg", "CameraReaderR_ball.csv", "Model3D.csv", "vicon.csv")
PINNED = {
    "001": "52de7c53efe21b0d728bbf1eecf854e109d7112bfbdad71ccfca9f1272a2e00c",
    "002": "2e0828ecd447dce1bf1ad03332f2bd06887f2859da6517f2d3a0179b5a5da9cd",
    "003": "ebf52d67ab7803e545e2369f2ee85a97f06fd7c33f507d09c20778b73fcb2a33",
    "docs": "fef473319ceb052e1eb3282255d8797e48d99989b963ed7f0de3346a9d4c7245",
}


def fingerprint(directory, names):
    sha = hashlib.sha256()
    for name in sorted(names):
        sha.update(name.encode("ascii") + b"\0" + hashlib.sha256((directory/name).read_bytes()).digest())
    return sha.hexdigest()


def check_files(out):
    for trial in TRIALS:
        if fingerprint(out/"vicon_dataset"/trial, NAMES) != PINNED[trial]:
            raise ValueError(f"Unexpected upstream content for trial {trial}")
    if fingerprint(out, ["README.md", "label_tool.py"]) != PINNED["docs"]:
        raise ValueError("Unexpected upstream documentation")


def get_range(spec):
    request = Request(f"{BASE}&chunk={spec}", headers={"Range": f"bytes={spec}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=90) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise ValueError("Public download did not serve a ZIP byte range. Supply --archive instead.")
        start, end, size = map(int, match.groups())
        expected = (size-int(spec[1:]), size-1) if spec.startswith("-") else tuple(map(int, spec.split("-")))
        if size != 644973627 or (start, end) != expected:
            raise ValueError(f"Unexpected range or upstream release: {response.headers.get('Content-Range')}")
        data = response.read(end-start+2)
        if len(data) != end-start+1:
            raise ValueError("Incomplete ZIP byte range")
        return data, start


def fetch(archive_path, out):
    try:
        check_files(out)
        return
    except (FileNotFoundError, ValueError):
        pass
    names = [f"vicon_dataset/{trial}/{name}" for trial in TRIALS for name in NAMES] + ["README.md", "label_tool.py"]
    window = None if archive_path else ArchiveWindow(*get_range("-524288"))
    with zipfile.ZipFile(archive_path or window) as archive:
        ordered = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
        following = [entry.header_offset for entry in ordered[1:]] + [archive.start_dir]
        spans = {entry.filename: (entry.header_offset, end) for entry, end in zip(ordered, following)}
        groups = []
        for name in sorted(names, key=lambda n: spans[n][0]):
            start, end = spans[name]
            if groups and start-groups[-1][1] < 4096:
                groups[-1][1] = end
                groups[-1][2].append(name)
            else:
                groups.append([start, end, [name]])
        for start, end, group in groups:
            if window is not None:
                window.load(*get_range(f"{start}-{end-1}"))
            for name in group:
                data = archive.read(name)  # verify ZIP member CRC before publishing
                path = out / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    check_files(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="official ZIP for offline extraction")
    parser.add_argument("--out", type=Path, default=RAW)
    args = parser.parse_args()
    fetch(args.archive, args.out)
