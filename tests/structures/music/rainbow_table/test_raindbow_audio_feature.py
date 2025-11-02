def test_rainbow_audio_feature_importable():
    import importlib
    mod = importlib.import_module('app.structures.music.rainbow_table.raindbow_audio_feature')
    assert mod is not None
    # Module should have a file and expose at least one public attribute
    assert getattr(mod, '__file__', None)
    public = [n for n in dir(mod) if not n.startswith('_')]
    assert len(public) >= 1

