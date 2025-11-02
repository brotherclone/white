def test_book_chain_artifact_importable():
    import importlib
    candidates = [
        'app.structures.artifacts.book_chain_artifact',
        'app.structures.artifacts.book_chain'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None

