from pathlib import Path


def get_project_root() -> str:
    return str(Path(__file__).parent.parent.parent)


def list_files(dirpath: str, pattern: str) -> list:
    """List files in a directory
    eg. of pattern: '*.csv' """
    return list(Path(dirpath).glob(pattern))
