def test_text_chain_artifact_file_importable():
    import importlib
    candidates = [
        'app.structures.artifacts.text_chain_artifact_file',
        'app.structures.artifacts.text_chain_artifact'
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

