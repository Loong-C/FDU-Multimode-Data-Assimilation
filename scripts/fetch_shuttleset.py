"""Fetch the pinned stroke annotations and court metadata for the Fuzhou match."""

import argparse
from pathlib import Path
from download_sample import fetch

REVISION = "45517f7d4cb936b03f3eabf939cc7959d39226fe"
BASE = f"https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/{REVISION}"
MATCH = "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals"
FILES = {
    f"ShuttleSet/set/{MATCH}/set1.csv": "e3a5670375aef168e6fa006d3de43d74e936f11d2e54c144603ae863c45502e0",
    f"ShuttleSet/set/{MATCH}/set2.csv": "e41592803d405a79757f1d620a5f715fc070114bcf15a1b890b57cc3fce2cd5e",
    f"ShuttleSet/set/{MATCH}/set3.csv": "856d0fd18657a94e09f6563690df7aae301a0ae8ec45745112419ceb89ae4b8a",
    "ShuttleSet/set/match.csv": "e0a07f6f3dead61fdef27be15acf78e9298d4913bc4718b6fc7d0c9c55ebf6ac",
    "ShuttleSet/set/homography.csv": "dd529642da81963f6c2d9a59da55ad8be6d4779ef92cd79ca6910e87f2eeaf1e",
    "LICENSE": "2ff939aee5c1dbce2051c009933dc0ca17f75e75b26c844b73780f2498b33164",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/shuttleset")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for remote, digest in FILES.items():
        fetch(args.out / Path(remote).name, f"{BASE}/{remote}", digest)
    print(f"Verified {len(FILES)} ShuttleSet files in {args.out}")


if __name__ == "__main__":
    main()
