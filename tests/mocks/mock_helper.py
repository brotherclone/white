import os
import builtins

from pathlib import Path

def redirect_test_mocks_open(monkeypatch):
    real_open = builtins.open
    env_path = os.getenv("AGENT_MOCK_DATA_PATH")
    if env_path:
        mock_root = Path(env_path)
    else:
        resolved = Path(__file__).resolve()
        repo_root = None
        for p in resolved.parents:
            if (p / "pyproject.toml").exists() or (p / "README.md").exists():
                repo_root = p
                break
        if repo_root is None:
            repo_root = resolved.parents[2]
        mock_root = repo_root / "tests" / "mocks"

    def _open(path, mode='r', *args, **kwargs):
        if isinstance(path, str) and path.startswith("/tests/mocks/"):
            rel = path[len("/tests/mocks/"):]
            real_path = (mock_root / rel).resolve()
            return real_open(real_path, mode, *args, **kwargs)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open, raising=False)

