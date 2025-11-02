def test_earthly_frames_mcp_importable():
    import importlib
    candidates = [
        'app.reference.mcp.earthly_frames_mcp',
        'app.reference.mcp.earthly_frames'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None

