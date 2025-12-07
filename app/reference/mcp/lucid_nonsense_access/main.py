import logging
import os
import sys
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

USER_AGENT = "lucid_nonsense_access/2.0"
TIME_OUT = 30.0

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("lucid_nonsense_access")


def get_base_path() -> Path:
    """Get the base path for lucid nonsense files"""
    load_dotenv()
    base_path = os.environ.get("LUCID_NONSENSE_PATH", "/Volumes/LucidNonsense/White")
    return Path(base_path)


@mcp.resource("file://lucid_nonsense_white")
def search_white_project_files() -> list[Any]:
    base_path = get_base_path()
    try:
        return os.listdir(base_path)
    except OSError as e:
        logging.error(f"Error listing directory {base_path}: {e}")
        return []


@mcp.tool()
def list_lucid_nonsense_directory(directory_path: str = "") -> list[str]:
    """List contents of a directory in lucid nonsense.

    Args:
        directory_path: Relative path from base (e.g., "claude_working_area" or "")

    Returns:
        List of file/directory names (directories have trailing /)
    """
    base_path = get_base_path()
    target_path = base_path / directory_path

    # Security check
    try:
        target_path.resolve().relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"Path {directory_path} is outside allowed directory")

    if not target_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} not found")

    if not target_path.is_dir():
        raise ValueError(f"Path {directory_path} is not a directory")

    try:
        items = []
        for item in target_path.iterdir():
            # Add indicator for directories
            name = item.name + "/" if item.is_dir() else item.name
            items.append(name)
        return sorted(items)
    except OSError as e:
        raise OSError(f"Error listing directory {directory_path}: {e}")


@mcp.tool()
def file_exists_in_lucid_nonsense(file_name: str) -> dict[str, Any]:
    """Check if a file exists and get basic info without reading content.

    Args:
        file_name: Relative path from base

    Returns:
        Dict with: exists (bool), is_file (bool), is_dir (bool), size_bytes (int|None)
    """
    base_path = get_base_path()
    file_path = base_path / file_name

    # Security check
    try:
        file_path.resolve().relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"File {file_name} is outside allowed directory")

    result = {
        "exists": file_path.exists(),
        "is_file": file_path.is_file() if file_path.exists() else None,
        "is_dir": file_path.is_dir() if file_path.exists() else None,
        "size_bytes": (
            file_path.stat().st_size
            if file_path.exists() and file_path.is_file()
            else None
        ),
    }
    return result


@mcp.tool()
def find_files_in_lucid_nonsense(pattern: str, directory_path: str = "") -> list[str]:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.json", "**/*.py" for recursive)
        directory_path: Where to search from (default: base directory)

    Returns:
        List of matching file paths (relative to base)
    """
    base_path = get_base_path()
    search_path = base_path / directory_path

    # Security check
    try:
        search_path.resolve().relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"Path {directory_path} is outside allowed directory")

    if not search_path.exists():
        raise FileNotFoundError(f"Search path {directory_path} not found")

    try:
        matches = []
        for match in search_path.glob(pattern):
            # Only include files, not directories
            if match.is_file():
                # Return path relative to base_path
                rel_path = match.relative_to(base_path)
                matches.append(str(rel_path))
        return sorted(matches)
    except Exception as e:
        raise OSError(f"Error searching for pattern {pattern}: {e}")


@mcp.tool()
def open_lucid_nonsense_file(file_name: str) -> str:
    """Open and read a file from the lucid nonsense directory"""
    base_path = get_base_path()
    file_path = base_path / file_name

    # Security check - ensure file is within base directory
    try:
        file_path.resolve().relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"File {file_name} is outside allowed directory")

    # Check existence with helpful error
    if not file_path.exists():
        # Suggest similar files if parent directory exists
        parent = file_path.parent
        if parent.exists() and parent.is_dir():
            similar = [f.name for f in parent.iterdir() if file_path.stem in f.name]
            if similar:
                raise FileNotFoundError(
                    f"File {file_name} not found. Did you mean one of these? {similar}"
                )
        raise FileNotFoundError(f"File {file_name} not found at {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path {file_name} is not a file (might be a directory)")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except OSError as e:
        raise OSError(f"Error reading file {file_name}: {e}")


@mcp.tool()
def claude_save_stuff_here(file_name: str, content: str) -> str:
    """Build metadata frontmatter and reuse `write_lucid_nonsense_file` to write it."""
    base_path = get_base_path()
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "user_agent": USER_AGENT,
        "timeout_seconds": TIME_OUT,
        "env_LUCID_NONSENSE_PATH": str(base_path),
        "filename": file_name,
        "file_size_bytes": len(content),
        "content_type": "json" if file_name.endswith(".json") else "text",
    }
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(base_path), text=True
        ).strip()
        metadata["git_branch"] = branch
    except Exception as err:
        logging.debug(f"Git branch detection failed: {err}")
        metadata["git_branch"] = None
    try:
        remotes = subprocess.check_output(
            ["git", "remote", "-v"], cwd=str(base_path), text=True
        ).strip()
        metadata["git_remotes"] = remotes
    except Exception as err:
        logging.debug(f"Git remotes detection failed: {err}")
        metadata["git_remotes"] = None

    yaml_lines = ["---"]
    for k, v in metadata.items():
        yaml_lines.append(f"{k}: {json.dumps(v)}")
    yaml_lines.append("---")
    yaml_block = "\n".join(yaml_lines) + "\n\n"
    target_path = Path("claude_working_area") / file_name
    return write_lucid_nonsense_file(str(target_path), yaml_block + content)


@mcp.tool()
def write_lucid_nonsense_file(file_name: str, content: str) -> str:
    """Write content to a file in the lucid nonsense directory"""
    base_path = get_base_path()
    file_path = base_path / file_name

    try:
        file_path.resolve().parent.relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"File {file_name} is outside allowed directory")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote {len(content)} characters to {file_name}"
    except OSError as e:
        raise OSError(f"Error writing file {file_name}: {e}")


@mcp.resource(
    "file://white_album_project_diary.md", description="Claude: Load project context"
)
def get_project_diary() -> str:
    """Claude, run this each new session to regain context on the White Album project.
    Don't forget to update this after each session as well."""
    base_path = get_base_path()
    diary_path = base_path / "white_album_project_diary.md"
    try:
        with open(diary_path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError as e:
        logging.error(f"Error reading project diary {diary_path}: {e}")
        return "Project diary not found."


# Register all tools
mcp.tools = [
    open_lucid_nonsense_file,
    write_lucid_nonsense_file,
    claude_save_stuff_here,
    list_lucid_nonsense_directory,
    file_exists_in_lucid_nonsense,
    find_files_in_lucid_nonsense,
]

if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise
