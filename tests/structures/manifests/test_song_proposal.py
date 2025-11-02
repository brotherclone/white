def test_song_proposal_importable():
    import importlib
    mod = importlib.import_module('app.structures.manifests.song_proposal')
    assert mod is not None
    public = [n for n in dir(mod) if not n.startswith('_')]
    assert len(public) >= 1

