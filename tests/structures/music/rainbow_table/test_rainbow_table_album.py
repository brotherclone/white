def test_rainbow_table_album_importable():
    import importlib
    candidates = [
        'app.structures.music.rainbow_table.rainbow_table_album',
        'app.structures.music.rainbow_table.album'
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

