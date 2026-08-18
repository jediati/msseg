"""File discovery, image loading and CSV plumbing shared by the measure scripts.

Every ``measure_*`` / ``calculate_*`` / ``plot_*`` script under
``msseg/mscoupon/`` used to carry its own copy of natural sorting, TIFF
discovery, ``START:END`` parsing and the ``--append`` CSV idiom -- seven copies
of some of them. They live here once.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TIFF_SUFFIXES = (".tif", ".tiff")


def natural_sort_key(path) -> List:
    """Sort key that orders ``recon_2.tif`` before ``recon_10.tif``.

    Accepts a ``Path`` or a string.
    """
    name = path.name if isinstance(path, Path) else str(path)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def find_tiff_files(folder) -> List[Path]:
    """Return the folder's TIFFs in natural filename order."""
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES]
    return sorted(files, key=natural_sort_key)


def parse_range(text: str) -> Tuple[int, int]:
    """Parse a required ``START:END`` pair (both ends mandatory, END > START)."""
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected START:END, got {text}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Expected integer START:END, got {text}") from None
    if start < 0 or end < 0:
        raise ValueError("Coordinates must be >= 0")
    if end <= start:
        raise ValueError("END must be greater than START")
    return start, end


def parse_samples(spec: Optional[str], n_files: int) -> Tuple[int, int]:
    """Parse an optional ``START:END`` image subsequence.

    Python-style indexing (START inclusive, END exclusive) and either end may be
    omitted: ``:100``, ``2500:``, ``100:110``. ``None`` selects everything.
    Indices are clamped to ``n_files``.
    """
    if spec is None:
        return 0, n_files

    parts = str(spec).split(":")
    if len(parts) != 2:
        raise ValueError("--samples must have format START:END")

    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else n_files
    if start < 0 or end < 0:
        raise ValueError("--samples indices must be >= 0")
    if start > end:
        raise ValueError("--samples START must be <= END")
    return min(start, n_files), min(end, n_files)


def read_tiff(path):
    """Read one TIFF as a numpy array (imported lazily so this module stays light)."""
    import tifffile
    return tifffile.imread(str(path))


def format_float(value: Any) -> Any:
    """Render floats at the 12-significant-digit precision the CSVs use."""
    return f"{value:.12g}" if isinstance(value, float) else value


def format_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply :func:`format_float` to every value of a CSV row."""
    return {k: format_float(v) for k, v in row.items()}


class CsvWriter:
    """A ``csv.DictWriter`` that understands the scripts' ``--append`` contract.

    The header is written only when the file is new or empty, floats are
    formatted consistently, and each row is flushed so a long run can be
    inspected (or interrupted) without losing what it has already measured.
    """

    def __init__(self, path, fieldnames: Sequence[str], append: bool = False):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        write_header = True
        mode = "w"
        if append:
            mode = "a"
            write_header = not self.path.exists() or self.path.stat().st_size == 0

        self._fh = self.path.open(mode, newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(format_row(row))
        self._fh.flush()

    def write_blank(self, **known: Any) -> None:
        """Write an all-empty row carrying only the identifying columns.

        Used when a slice fails: the run keeps going and the CSV still has one
        row per image, so row indices line up with the input sequence.
        """
        row = {name: "" for name in self.fieldnames}
        row.update(known)
        self.write(row)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def select_files(folder, samples: Optional[str] = None) -> Tuple[List[Path], int, int]:
    """Resolve a folder + ``--samples`` spec to (selected files, start, end).

    Raises ``FileNotFoundError`` / ``ValueError`` with the messages the scripts
    print, so each CLI can report and exit without repeating the checks.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    files = find_tiff_files(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {folder}")

    start, end = parse_samples(samples, len(files))
    selected = files[start:end]
    if not selected:
        raise ValueError("Selected subsequence contains no TIFF files.")
    return selected, start, end


def iter_images(files: Iterable[Path], start: int = 0):
    """Yield ``(local_index, image_index, path, image)`` for each file.

    The image is read lazily and dropped after each iteration, so a whole stack
    never sits in memory at once.
    """
    for local_i, path in enumerate(files):
        yield local_i, start + local_i, path, read_tiff(path)
