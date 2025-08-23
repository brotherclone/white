import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from typing import Any
from mcp.server.fastmcp import FastMCP

USER_AGENT = "lucid_nonsense_access/1.0"
TIME_OUT = 30.0

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("lucid_nonsense_access")


def get_base_path() -> Path:
    """Get the base path for lucid nonsense files"""
    load_dotenv()
    base_path = os.environ.get('LUCID_NONSENSE_PATH', '/Volumes/LucidNonsense/White')
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
def open_lucid_nonsense_file(file_name: str) -> str:
    """Open and read a file from the lucid nonsense directory"""
    base_path = get_base_path()
    file_path = base_path / file_name

    # Security check - ensure file is within base directory
    try:
        file_path.resolve().relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"File {file_name} is outside allowed directory")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except OSError as e:
        raise OSError(f"Error reading file {file_name}: {e}")


@mcp.tool()
def write_lucid_nonsense_file(file_name: str, content: str) -> str:
    """Write content to a file in the lucid nonsense directory"""
    base_path = get_base_path()
    file_path = base_path / file_name

    # Security check - ensure file is within base directory
    try:
        file_path.resolve().parent.relative_to(base_path.resolve())
    except ValueError:
        raise ValueError(f"File {file_name} is outside allowed directory")

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote {len(content)} characters to {file_name}"
    except OSError as e:
        raise OSError(f"Error writing file {file_name}: {e}")

@mcp.resource("file://white_album_project_diary.md")
def get_project_diary() -> str:
    """Claude, run this each new session to regain context on the White Album project.
    Don't forget to update this after each session as well."""
    base_path = get_base_path()
    diary_path = base_path / "white_album_project_diary.md"
    try:
        with open(diary_path, 'r', encoding='utf-8') as file:
            return file.read()
    except OSError as e:
        logging.error(f"Error reading project diary {diary_path}: {e}")
        return "Project diary not found."

mcp.tools = [
    search_white_project_files,
    open_lucid_nonsense_file,
    write_lucid_nonsense_file
]

if __name__ == "__main__":
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise