"""Walking an untrusted corpus, and knowing which document is the current one.

The graded corpus contains superseded documents next to their replacements (a
``..._r1_WITHDRAWN.pdf`` beside ``..._r2.pdf``) and expects the *current* one to be
cited. Revision awareness therefore lives here, computed once at index time from both
the file name and the file's own content.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DocRecord",
    "corpus_fingerprint",
    "family_and_revision",
    "mark_superseded",
    "walk_corpus",
]

IGNORED_DIRS = frozenset(
    {"__pycache__", ".git", ".svn", ".hg", "node_modules", ".ipynb_checkpoints"}
)
_STATUS = re.compile(
    r"[_\-. ]*(withdrawn|superseded|obsolete|deprecated|draft|archived?|old)", re.I
)
_REVISION = re.compile(
    r"[_\-. ]+(?:rev(?:ision)?|r|v|version)[_\-. ]?([0-9]+[a-z]?[0-9]*|[a-z][0-9]*)$", re.I
)


@dataclass
class DocRecord:
    """Everything the index knows about one file of the corpus."""

    path: str  # corpus-relative, forward slashes: exactly what a citation must contain
    kind: str | None
    indexed: bool = False
    reason: str | None = None
    withdrawn: bool = False
    family: str = ""
    revision: tuple[int, ...] = (0,)
    superseded_by: str | None = None
    text: str = field(default="", repr=False)

    @property
    def superseded(self) -> bool:
        """True when this document is withdrawn or a newer revision of it exists."""
        return self.withdrawn or self.superseded_by is not None


def walk_corpus(root: Path) -> Iterator[Path]:
    """Yield every file under ``root`` in a stable order, tolerating unreadable directories.

    ``os.walk`` swallows a listing error by default; the callback makes that explicit.
    A directory we may not list is simply skipped (its files cannot be answers), and an
    empty directory yields nothing.
    """
    for directory, subdirs, files in os.walk(root, onerror=lambda _error: None):
        subdirs[:] = sorted(d for d in subdirs if d not in IGNORED_DIRS)
        for name in sorted(files):
            yield Path(directory) / name


def corpus_fingerprint(root: Path) -> str:
    """Cheap identity of a corpus: file names and sizes (not mtimes, which can drift)."""
    import hashlib

    digest = hashlib.sha1(usedforsecurity=False)
    for path in walk_corpus(root):
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        digest.update(f"{path.relative_to(root).as_posix()}:{size}\n".encode())
    return digest.hexdigest()


def _revision_key(token: str) -> tuple[int, ...]:
    letters = re.sub(r"[^a-z]", "", token.lower())
    digits = tuple(int(d) for d in re.findall(r"\d+", token))
    return (sum(ord(c) - 96 for c in letters), *digits) if letters else (0, *digits)


def family_and_revision(path: str) -> tuple[str, tuple[int, ...], bool]:
    """Split a file name into (document family, revision, marked-withdrawn-by-name).

    ``specs/tq40_datasheet_r1_WITHDRAWN.pdf`` -> (``specs/tq40_datasheet``, (0, 1), True).
    """
    posix = Path(path)
    stem = posix.stem
    name_withdrawn = bool(_STATUS.search(stem))
    stem = _STATUS.sub("", stem)
    revision: tuple[int, ...] = (0,)
    match = _REVISION.search(stem)
    if match:
        revision = _revision_key(match.group(1))
        stem = stem[: match.start()]
    family = (posix.parent / stem.lower().strip("_-. ")).as_posix()
    return family, revision, name_withdrawn


def mark_superseded(records: list[DocRecord]) -> None:
    """Fill in ``superseded_by`` for every document that has a newer, current sibling."""
    families: dict[str, list[DocRecord]] = defaultdict(list)
    for record in records:
        if record.indexed:
            families[record.family].append(record)
    for members in families.values():
        current = [m for m in members if not m.withdrawn]
        if not current:
            continue
        newest = max(current, key=lambda m: m.revision)
        for member in members:
            if member is not newest and (member.withdrawn or member.revision < newest.revision):
                member.superseded_by = newest.path
