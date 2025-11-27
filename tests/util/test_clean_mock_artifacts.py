from pathlib import Path

from app.util.clean_mock_artifacts import find_mock_dirs, delete_paths, main


def test_find_mock_dirs_top_level_only(tmp_path: Path):
    # create top-level dirs
    (tmp_path / "mock_one").mkdir()
    (tmp_path / "AnotherMock").mkdir()
    (tmp_path / "unrelated").mkdir()

    # nested mock should not be discovered because find_mock_dirs only inspects top-level
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "mock_nested").mkdir()

    results = find_mock_dirs(tmp_path)
    names = sorted(p.name for p in results)
    assert names == ["AnotherMock", "mock_one"]


def test_delete_paths_handles_files_dirs_and_symlinks(tmp_path: Path):
    # create a file
    f = tmp_path / "file.txt"
    f.write_text("hello")

    # create a directory with a file inside
    d = tmp_path / "adir"
    d.mkdir()
    (d / "inner.txt").write_text("x")

    # create a symlink pointing to the file
    link = tmp_path / "link_to_file"
    try:
        link.symlink_to(f)
    except (NotImplementedError, OSError):
        # some filesystems/CI images don't allow symlinks; skip symlink in that case
        link = None

    paths = [f, d]
    if link is not None:
        paths.append(link)

    deleted = delete_paths(paths, verbose=False)
    assert deleted == len(paths)

    for p in paths:
        assert not p.exists()


def test_main_dry_run_and_force_delete(tmp_path: Path):
    base = tmp_path / "chain_artifacts"
    base.mkdir()
    (base / "keep_dir").mkdir()
    mock_dir = base / "something_mock"
    mock_dir.mkdir()

    # dry run should not delete
    rc = main(["--base", str(base), "--dry-run"])  # returns exit code 0
    assert rc == 0
    assert mock_dir.exists()

    # forced delete should remove the mock dir
    rc2 = main(["--base", str(base), "-y"])  # -y to skip prompt
    assert rc2 == 0
    assert not mock_dir.exists()
    # ensure non-mock dir remains
    assert (base / "keep_dir").exists()
