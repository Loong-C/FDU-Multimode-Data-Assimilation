"""Fetch the eight released Fuzhou rallies and draw a contact sheet of raw frames."""
import argparse
import hashlib
import io
import re
import tempfile
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/"
    "EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g?download=1"
)
# SHA-256 of the uncompressed MP4 and CSV released for each rally, in that order.
MATCH4 = {
    "1_03_02": ("a1a98a4a7a0a07f1b63ed3bbbdb4a7e812743ce3852415af0a261d2ae253e304",
                "fcf69412bdbf5aab15d54c07c44fe964e51796114b860d3dbbb6402d14b28c74"),
    "1_15_10": ("01f29220088f5f31ee10aa7bd9fa7097b0c8cb7fcc9c2be2e7f3274f24ba2004",
                "3b4610ff23ca0efbfaaa80cfb4a9639c8391817b9b7e78af780ad753dfabbdc9"),
    "2_02_05": ("608aa78c323fac1441af223d381334e7778915e276c0da4644bcaa95aa3a6995",
                "61a7e42805369af3738c4043fa9904d28731d6fe9f2bb20c3aba0af0c76bbde0"),
    "2_05_07": ("e9e034131d0e33291a064c20c976a60cca0356c865b09b1c7fcbf6d65841ae72",
                "9ef062a168cc48d88c62a328886647e785ad6680f3faa6712f93c967ce9e12e6"),
    "2_14_17": ("806168aa3544395ac337faf86dd07f718795785dd78f1716cd10ef3cc22a2876",
                "4dadb91d9ac6da3adc31bdcc5376f570e4a6d4f11eae4364255c028ee80837d2"),
    "3_02_00": ("439a0af8cb29ea50e249cdc9935850f1e25010f2893d6725055baaabd16cf658",
                "11e9150aeaec9c67a3d97278781777f8fbc7b0c52b55e705116cf6b0af81788c"),
    "3_07_05": ("949ff5be9c3605e0cd6b5b455acd68935d0b50f9d734a0b5daa0ec80051d461b",
                "cd760f8b53017a158f7d437723a0281bbc42ea80894e7ca052657769f2ca8eb6"),
    "3_18_17": ("828b5b89224a242c36b4c6c5d7618a29384e8c7d59769f35e9569df34ada42db",
                "dd100659cb7925a2ca7303f3e5f54331d603a6f80958ff887842138366cdc348"),
}


class ArchiveWindow(io.BytesIO):
    """Expose a downloaded byte range at the same offsets as the remote ZIP."""

    def __init__(self, data, start):
        super().__init__(data)
        self.start = start

    def tell(self):
        return self.start + super().tell()

    def seek(self, offset, whence=0):
        if whence == 0:
            offset -= self.start
        return self.start + super().seek(offset, whence)

    def load(self, data, start):
        super().seek(0)
        self.truncate(0)
        self.write(data)
        self.start = start


def get_range(spec):
    request = Request(URL, headers={"Range": f"bytes={spec}", "Accept-Encoding": "identity"})
    # SharePoint sets a cookie on the public share link before redirecting.
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    with opener.open(request, timeout=100) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise ValueError("Server did not return a ZIP byte range. Supply --archive for offline use.")
        start, end, total = map(int, match.groups())
        if spec.startswith("-"):
            expected = max(0, total - int(spec[1:])), total - 1
        else:
            expected = tuple(map(int, spec.split("-")))
        if (start, end) != expected:
            raise ValueError(f"Range mismatch: requested {spec}, received {start}-{end}")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError(f"Incomplete response for range {spec}")
    return data, start


def save_member(archive, name, destination, digest):
    data = archive.read(name)  # zipfile also checks CRC-32.
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        tmp.write(data)
    temporary = Path(tmp.name)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_members(files, archive=None):
    """Fetch {member: (destination Path, SHA-256)} from the official dataset ZIP."""
    pending = {name: (path, digest) for name, (path, digest) in files.items()
               if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest}
    if not pending:
        return
    if archive is not None:
        with zipfile.ZipFile(archive) as dataset:
            for name, (path, digest) in pending.items():
                save_member(dataset, name, path, digest)
        return

    # The published central directory is 44,934 bytes. Read it once, then
    # coalesce adjacent member records and let zipfile decode the original data.
    window = ArchiveWindow(*get_range("-131072"))
    with zipfile.ZipFile(window) as dataset:
        entries = sorted(dataset.infolist(), key=lambda item: item.header_offset)
        following = [item.header_offset for item in entries[1:]] + [dataset.start_dir]
        spans = {item.filename: (item.header_offset, end) for item, end in zip(entries, following)}
        groups = []
        for name in sorted(pending, key=lambda n: spans[n][0]):
            start, stop = spans[name]
            if groups and start - groups[-1][1] <= 65536:
                groups[-1][1] = stop
                groups[-1][2].append(name)
            else:
                groups.append([start, stop, [name]])
        for start, stop, names in groups:
            window.load(*get_range(f"{start}-{stop - 1}"))
            for name in names:
                save_member(dataset, name, *pending[name])


def preview(match, output, *, frames=(("1_03_02", 100), ("2_14_17", 82), ("3_18_17", 275), ("1_15_10", 220))):
    import cv2

    panels = []
    for rally, frame in frames:
        cap = cv2.VideoCapture(str(match / "video" / f"{rally}.mp4"))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Could not decode {rally} frame {frame}")
            seconds = frame / cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        panel = cv2.resize(image, (608, 342), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(panel, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        caption = f"{rally}   |   Frame {frame}   |   {seconds:.2f} s"
        cv2.putText(panel, caption, (12, 364), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = cv2.vconcat([cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])])
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="optional local TrackNetV2.zip")
    args = parser.parse_args()
    match = ROOT / "data/raw/tracknetv3/Professional/match4"
    files = {}
    for rally, (video_sha, csv_sha) in MATCH4.items():
        for relative, digest in [(f"video/{rally}.mp4", video_sha), (f"csv/{rally}_ball.csv", csv_sha)]:
            files[f"TrackNetV2/Professional/match4/{relative}"] = (match / relative, digest)
    fetch_members(files, args.archive)
    preview(match, ROOT / "artifacts/previews/tracknetv3.jpg")


if __name__ == "__main__":
    main()
