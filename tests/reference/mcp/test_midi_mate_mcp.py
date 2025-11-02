def test_mcp_midi_mate_importable():
    import importlib
    candidates = [
        'app.reference.mcp.midi_mate_mcp',
        'app.reference.mcp.midi_mate'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None

