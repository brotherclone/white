def safe_add(x, y):
    """Safely add two lists, handling None values"""
    if x is None and y is None:
        return None
    if x is None:
        return y
    if y is None:
        return x
    return x + y
