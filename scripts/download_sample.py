"""Cache individual source files only after checking their recorded SHA-256."""
import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


def fetch(path, url, expected_hash):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with urlopen(url, timeout=120) as source, temp.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Unexpected SHA-256 for {path.name}. Check the upstream release.")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
