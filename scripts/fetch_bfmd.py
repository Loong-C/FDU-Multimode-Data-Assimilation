"""Download the BFMD study sample and build a four-tournament contact sheet."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from download_sample import fetch


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bfmd"
PREVIEW = ROOT / "artifacts" / "previews" / "bfmd.jpg"
CLIP_BASE = "https://ning-d.github.io/BFMD-Dataset/videos"
CLIPS = [
    # File tag, tournament, local zero-based frame number, SHA-256.
    ("kapal", "Indonesia Open", 286, "1f784dd3ea1442b86cb619a9fddb88fa68d06713e8b59a43e10bc65f1e2c3f7d"),
    ("petronas", "Malaysia Open", 45, "4d9a6c589776490e9e9245bf6bf1c55ab479125d2eac31fff0d58e28aaf463e0"),
    ("victor", "China Open", 45, "7f3d0d3521e08125971bfd52532cc52a17bdc2ab8823e47fc5fd0f3a9eaaf23c"),
    ("yonex", "All England Open", 120, "65ab1f1d772bafffd81d5b45831538beebb49376370bea72a7557ea6625c7f53"),
]

# Stable IDs in the public BFMD Drive, and hashes of the files actually inspected.
DRIVE_FILES = [
    ("Badminton_video_list.csv", "1bG5ZIjfE0CWyAnGqPdpq7u0s8Ygzz7OM", "af0d3ab37965759af62ea7fbf64fa35638e4251d6464e37ae0e086834df61398"),
    ("match_index.csv", "1EzJ3khkIBr9aKDQm6DUcTGvqqnFAyUaX", "cf740ac2632ed52ba75b663e0698cd926357e230e94985a1edca5d92ee165fd1"),
    ("malaysia/metadata.json", "1hyS5jmIl3saX4iyLyTNla2T4OlPfY_rk", "c96a89f19a0f2c43c76dd4d43f26ceb9d676bb907db3adf5ddf9e723449620d7"),
    ("malaysia/court.json", "1aLfkyWrAHplR6SWdSton7PqDUnrrzlaj", "6a62e0e7bb3d59fac803741f5b2c52170d7fa1d2e57450541100973af9060149"),
    ("malaysia/court_perframe.json", "1ARYpYok-aO5s3dD3G1fvrx63xyusF66w", "9b4d0596df851a7cccc1487765c59aec87885ef2be4a038ba645bb470d070688"),
    ("malaysia/shuttle.json", "1N_dmk0iStuoX0hElU8jSXGp_QL5s7znU", "4e66a575687bb4a6d2a8c9f37e837226551d9980e7d8ed7e25b208bda80e4026"),
    ("malaysia/shot_type.json", "1jvcUGjoFtnbPD6JV41P5BqeAvrVFVr8x", "c22ca7698c500976fa9589038c2d101eb7927be2a95250d0f54b073fee94ebbd"),
    ("malaysia/hit_inferred.json", "1Cof3XgHU7W14YjCewdRlYILU_PNZJPPQ", "f22fc3bb3e5309cae851511426990cb0d8c6718c6b6c9b4e25aa142a1772cd8d"),
    ("malaysia/pose.json", "11alosIr7znUiPOfw53wlPNZD-neTaYc_", "635deb793642d00bdfe2c6f4cf2628663bf17e42f3ce5557651fe3d54546dfe4"),
    ("malaysia/player_bbox.json", "100v4gznBeBvHLU0hiig4x1c-KKBUk6ot", "49b7dfb84260b6d81a63a9171e4c5a00b61159ee25838adc7b0332c8501c4e8b"),
    ("malaysia/caption.json", "1K6rcxDdXP9l97s-o52eUpBE8Mw4RdUv5", "c68847d4922a6eb2ad93e6aef900aada0cfb1987067fdc86f9ae428d6cd8f512"),
]


def make_preview():
    panels = []
    for tag, event, frame_number, expected_hash in CLIPS:
        filename = f"demo_{tag}.mp4"
        path = RAW / filename
        fetch(path, f"{CLIP_BASE}/{filename}", expected_hash)
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not ok or fps <= 0:
                raise OSError(f"Cannot decode frame {frame_number} of {path}")
        finally:
            capture.release()
        frame = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(27, 28, 30))
        caption = f"{event} 2025   |   frame {frame_number}   |   {frame_number / fps:.2f} s"
        cv2.putText(panel, caption, (14, 386), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(panel)
    sheet = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(PREVIEW), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise OSError(f"Cannot write {PREVIEW}")
    print(PREVIEW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true", help="skip the Malaysia match annotations")
    args = parser.parse_args()
    make_preview()
    if not args.preview_only:
        for filename, file_id, expected_hash in DRIVE_FILES:
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            fetch(RAW / filename, url, expected_hash)
        print(f"Verified {len(DRIVE_FILES)} BFMD annotation and index files in {RAW}.")
