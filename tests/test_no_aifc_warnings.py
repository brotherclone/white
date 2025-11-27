import subprocess
import sys
from pathlib import Path


def test_run_entrypoint_no_aifc_warnings():
    """Run the CLI help and assert that aifc/sunau DeprecationWarnings are not on stderr."""
    project_root = Path.cwd()
    env = dict(**{**dict(**{}), **{}})
    # Ensure PYTHONPATH includes the project root so `sitecustomize.py` is loaded
    env.update({"PYTHONPATH": str(project_root)})

    proc = subprocess.run(
        [sys.executable, "run_white_agent.py", "--help"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    stderr = proc.stderr
    assert "aifc was removed" not in stderr
    assert "sunau was removed" not in stderr
    # if the process failed, surface output for debugging
    assert proc.returncode == 0
