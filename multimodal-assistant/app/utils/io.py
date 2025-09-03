from typing import Optional
from pathlib import Path

def save_bytes_to(path: str | Path, content: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path
