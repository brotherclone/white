def test_book_artifact_importable():
    import importlib
    candidates = [
        'app.structures.artifacts.book_artifact',
        'app.structures.artifacts.book'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None
    public = [n for n in dir(mod) if not n.startswith('_')]
    assert len(public) >= 1

