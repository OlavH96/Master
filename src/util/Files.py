import os
from pathlib import Path

def is_dir(path: str or Path) -> bool:

    return Path(path).is_dir()

def mkdir(path: str or Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.mkdir()
    return path


def cleardir(path: str or Path) -> str or Path:
    files = os.listdir(str(path))
    [os.remove(str(path / file)) for file in files]
    return path


def makedir_else_cleardir(path: str or Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.mkdir()
    else:
        # Delete existing files
        files = os.listdir(str(path))
        [os.remove(str(path / file)) for file in files]

    return path
