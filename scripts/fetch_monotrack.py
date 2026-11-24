"""Download three TrackNetV2 rallies used in MonoTrack and extract four preview frames."""

import argparse
from pathlib import Path

from fetch_tracknetv3 import ROOT, fetch_members, preview


SAMPLE = Path("TrackNetV2/Test/match3")
DIGESTS = {
    "csv/1_02_00_ball.csv": "30f63448c306ce36a1c49734df29297226a2f315fbecd5ce3158b87875952cd2",
    "csv/1_03_02_ball.csv": "5af70d7d3d47d4fb1e496aad816266109df81589f5452f5e94094c39cffea856",
    "csv/1_05_02_ball.csv": "cda34e4dee5b7c3c4eb5e66a0702ab478e4de1f881267bf8cd4f1d5902ce3f13",
    "video/1_02_00.mp4": "764947ee4424f82ea982fba454a15cca190e96d38fa7b78d9763d34b5775fb3c",
    "video/1_03_02.mp4": "014449255edc092a893ab99003cf2f4054923e990079264080f3c1646e4cfb4d",
    "video/1_05_02.mp4": "b7bdf045d61c147fa430cf6f5694515dd31f549a3eb4590b30b517cb5618d4e1",
}
PREVIEW_FRAMES = (("1_02_00", 300), ("1_03_02", 100), ("1_03_02", 410), ("1_05_02", 161))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/monotrack")
    parser.add_argument("--archive", type=Path, help="Read from a local official ZIP instead of the network.")
    parser.add_argument("--preview", type=Path, default=ROOT / "artifacts/previews/monotrack.jpg")
    args = parser.parse_args()
    members = {
        (SAMPLE / name).as_posix(): (args.out / SAMPLE / name, digest)
        for name, digest in DIGESTS.items()
    }
    fetch_members(members, archive=args.archive)
    preview(args.out / SAMPLE, args.preview, frames=PREVIEW_FRAMES)


if __name__ == "__main__":
    main()
