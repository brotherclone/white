def test_discogs_mcp_importable():
    import importlib
    candidates = [
        'app.reference.mcp.discogs_mcp',
        'app.reference.mcp.discogs'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None

