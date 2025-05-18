# classes.py
#
# A Class Collection Code for library building
# From MML Library by Nathmath

from pathlib import Path
import ast
from typing import Generator, List, Tuple


# Get top level classes defined in a file
def top_level_classes(py_file: str | Path) -> List[str]:
    """
    Return a list with the names of every top‑level `class` defined in
    the file at `py_file`.  Nested / inner classes are ignored.
    """
    py_file = Path(py_file).expanduser().resolve()

    if not py_file.is_file() or py_file.suffix != ".py":
        raise ValueError(f"{py_file} is not a valid .py file")

    source = py_file.read_text(encoding="utf-8")

    # Parse the file’s AST once
    tree = ast.parse(source, filename=str(py_file))

    # Keep only `ClassDef` nodes whose *direct* parent is the module itself.
    top_level = [
        node.name
        for node in tree.body               # direct children of the module
        if isinstance(node, ast.ClassDef)
    ]
    return top_level


# Iterate over files and yield them one by one
def iter_py_files(folder: str | Path, typ: str = "py", recursive: bool = True) -> Generator[Path, None, None]:
    """
    Yield every *.{typ} file in folder.

    Args
    ----
    folder      - directory to walk
    typ         - type of the file (defaule "py")
    recursive   - if True, walk sub‑directories too (default)
    """
    folder = Path(folder).expanduser().resolve()

    if recursive:
        yield from folder.rglob(f"*.{typ}")
    else:
        yield from folder.glob(f"*.{typ}")


# Example: Get py classes
if __name__ == "__main__":
    root = Path("./").expanduser()

    for py_path in iter_py_files(root):
        classes = top_level_classes(py_path)
        if classes:
            for cls in classes:
                print("'", cls, "'", ", ", sep = "", end = "")
