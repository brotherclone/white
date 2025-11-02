def test_rainbow_table_song_importable():
    import importlib
    # try a couple of plausible module names
    candidates = [
        'app.structures.music.rainbow_table.rainbow_table_song',
        'app.structures.music.rainbow_table.song'
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

