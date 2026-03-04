from __future__ import annotations

from pathlib import Path

from vibe.core.autocompletion.file_indexer.ignore_rules import IgnoreRules
from vibe.core.autocompletion.file_indexer.store import FileIndexStats, FileIndexStore
from vibe.core.autocompletion.file_indexer.watcher import Change


def _build_store() -> tuple[FileIndexStore, FileIndexStats]:
    stats = FileIndexStats()
    return FileIndexStore(ignore_rules=IgnoreRules(), stats=stats), stats


def test_snapshot_returns_cached_immutable_tuple(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")

    store, _ = _build_store()
    store.rebuild(tmp_path)

    first_snapshot = store.snapshot()
    second_snapshot = store.snapshot()

    assert isinstance(first_snapshot, tuple)
    assert first_snapshot is second_snapshot
    assert [entry.rel for entry in first_snapshot] == ["a.py", "b.py"]


def test_apply_changes_invalidates_snapshot_cache(tmp_path: Path) -> None:
    initial = tmp_path / "alpha.py"
    initial.write_text("", encoding="utf-8")

    store, stats = _build_store()
    store.rebuild(tmp_path)
    initial_snapshot = store.snapshot()

    added_file = tmp_path / "beta.py"
    added_file.write_text("", encoding="utf-8")
    store.apply_changes([(Change.added, added_file)])

    updated_snapshot = store.snapshot()

    assert updated_snapshot is not initial_snapshot
    assert [entry.rel for entry in updated_snapshot] == ["alpha.py", "beta.py"]
    assert stats.incremental_updates == 1


def test_apply_changes_removes_file_when_it_becomes_ignored(tmp_path: Path) -> None:
    tracked = tmp_path / "logs.txt"
    tracked.write_text("hello", encoding="utf-8")

    store, stats = _build_store()
    store.rebuild(tmp_path)
    assert [entry.rel for entry in store.snapshot()] == ["logs.txt"]

    tracked.write_text("hello", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.txt\n", encoding="utf-8")
    store._ignore_rules.ensure_for_root(tmp_path.resolve())
    store.apply_changes([(Change.modified, tracked)])

    assert store.snapshot() == ()
    assert stats.incremental_updates == 1


def test_apply_changes_removes_directory_when_it_becomes_ignored(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    tracked = docs_dir / "note.md"
    tracked.write_text("hello", encoding="utf-8")

    store, stats = _build_store()
    store.rebuild(tmp_path)
    assert [entry.rel for entry in store.snapshot()] == ["docs", "docs/note.md"]

    (tmp_path / ".gitignore").write_text("docs/\n", encoding="utf-8")
    store._ignore_rules.ensure_for_root(tmp_path.resolve())
    store.apply_changes([(Change.modified, docs_dir)])

    assert store.snapshot() == ()
    assert stats.incremental_updates == 1
