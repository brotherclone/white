def test_audio_chain_artifact_file_importable():
    import importlib
    candidates = [
        'app.structures.artifacts.audio_chain_artifact_file',
        'app.structures.artifacts.audio_chain_artifact'
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

