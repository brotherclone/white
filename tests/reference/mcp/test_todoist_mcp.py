def test_todoist_mcp_importable():
    import importlib
    candidates = [
        'app.reference.mcp.todoist_mcp',
        'app.reference.mcp.todoist'
    ]
    mod = None
    for c in candidates:
        try:
            mod = importlib.import_module(c)
            break
        except Exception:
            continue
    assert mod is not None

