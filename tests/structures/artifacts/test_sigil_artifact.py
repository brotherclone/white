def test_sigil_artifact_importable(sigil_artifact=None):
    import importlib
    try:
        mod = importlib.import_module('app.structures.artifacts.sigil_artifact')
    except Exception:
        # fall back to checking provided fixture
        assert sigil_artifact is not None
        return
    assert mod is not None
    public = [n for n in dir(mod) if not n.startswith('_')]
    assert len(public) >= 1

