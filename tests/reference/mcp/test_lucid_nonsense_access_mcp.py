def test_lucid_nonsense_access_mcp_importable():
    import importlib
    candidates = [
        'app.reference.mcp.lucid_nonsense_access_mcp',
        'app.reference.mcp.lucid_nonsense_access'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None

