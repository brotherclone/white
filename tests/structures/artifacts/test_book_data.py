def test_book_data_importable():
    import importlib
    candidates = [
        'app.structures.artifacts.book_data',
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
    assert getattr(mod, '__file__', None)

