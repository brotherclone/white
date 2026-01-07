import logging
import requests
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

USER_AGENT = "earthly_frames_todoist/1.2"
TIME_OUT = 30.0
EF_PROJECT_ID = "6CrfWqXrxppjhqMJ"

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()

mcp = FastMCP("earthly_frames_todoist")


@mcp.tool()
def get_earthly_frames_project_sections(
    project_id: str = EF_PROJECT_ID,
) -> List[Dict[str, Any]]:
    """
    Get all sections for the Earthly Frames Todoist project.

    Args:
        project_id: Todoist project ID (defaults to The Earthly Frames project)

    Returns:
        List of section dictionaries with id, name, project_id, order
    """
    try:
        # Use direct REST API to avoid paginator hanging issue
        token = os.environ.get("TODOIST_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.get(
            f"https://api.todoist.com/rest/v2/sections?project_id={project_id}",
            headers=headers,
            timeout=TIME_OUT,
        )
        response.raise_for_status()

        sections = response.json()
        return [
            {
                "id": s["id"],
                "name": s["name"],
                "project_id": s["project_id"],
                "order": s["order"],
            }
            for s in sections
        ]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(
                f"403 Forbidden: Access denied to project {project_id}. Check API token permissions."
            )
        elif e.response.status_code == 401:
            logger.error(
                f"401 Unauthorized: Invalid API token for project {project_id}."
            )
        else:
            logger.error(f"HTTP error fetching sections for project {project_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching sections for project {project_id}: {e}")
        return []


@mcp.tool()
def create_sigil_charging_task(
    sigil_description: str,
    charging_instructions: str,
    song_title: str,
    section_name: str = "Black Agent - Sigil Work",
) -> Dict[str, Any]:
    """
    Create a Todoist task for human to charge a sigil.

    Args:
        sigil_description: Description of the sigil glyph and its components
        charging_instructions: Ritual instructions for charging
        song_title: Title of the song this sigil is for
        section_name: Name of the Todoist section (defaults to Black Agent section)

    Returns:
        Task dictionary with id, content, url, project_id, section_id
        OR error dictionary with {"success": False, "error": "error message"}
    """
    try:
        token = os.environ.get("TODOIST_API_TOKEN")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Get sections using REST API
        sections_response = requests.get(
            f"https://api.todoist.com/rest/v2/sections?project_id={EF_PROJECT_ID}",
            headers=headers,
            timeout=TIME_OUT,
        )
        sections_response.raise_for_status()
        sections = sections_response.json()

        # Find or create the section
        section_id = None
        for s in sections:
            if s["name"] == section_name:
                section_id = s["id"]
                logger.info(f"Found existing section: {section_name} (id={section_id})")
                break

        if not section_id:
            # Create new section
            logger.info(f"Section '{section_name}' not found, attempting to create...")
            create_section_response = requests.post(
                "https://api.todoist.com/rest/v2/sections",
                headers=headers,
                json={"name": section_name, "project_id": EF_PROJECT_ID},
                timeout=TIME_OUT,
            )
            create_section_response.raise_for_status()
            new_section = create_section_response.json()
            section_id = new_section["id"]
            logger.info(f"Created new section: {section_name} (id={section_id})")

        task_content = f"🜏 Charge Sigil for '{song_title}'"
        task_description = f"""
**Sigil Glyph:**
{sigil_description}

**Charging Instructions:**
{charging_instructions}

**Song:** {song_title}

Mark this task complete after the sigil has been charged and released.
"""

        # Create task using REST API
        create_task_response = requests.post(
            "https://api.todoist.com/rest/v2/tasks",
            headers=headers,
            json={
                "content": task_content,
                "description": task_description,
                "project_id": EF_PROJECT_ID,
                "section_id": section_id,
                "priority": 3,
            },
            timeout=TIME_OUT,
        )
        create_task_response.raise_for_status()
        task = create_task_response.json()

        return {
            "success": True,
            "id": task["id"],
            "content": task["content"],
            "url": task["url"],
            "project_id": task["project_id"],
            "section_id": task.get("section_id"),
            "created_at": task["created_at"],
        }

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if e.response.status_code == 403:
            error_msg = f"403 Forbidden: Cannot create task in project {EF_PROJECT_ID}. Check API token permissions for project access and task creation."
            logger.error(error_msg)
        elif e.response.status_code == 401:
            error_msg = "401 Unauthorized: Invalid API token."
            logger.error(error_msg)
        else:
            error_msg = f"HTTP error creating sigil charging task: {e}"
            logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "status_code": e.response.status_code,
        }
    except Exception as e:
        error_msg = f"Unexpected error creating sigil charging task: {e}"
        logger.exception("Failed creating sigil charging task")
        return {"success": False, "error": error_msg}


@mcp.tool()
def list_pending_black_agent_tasks(
    section_name: str = "Black Agent - Sigil Work",
) -> List[Dict[str, Any]]:
    """
    List all incomplete tasks in Black Agent sections.

    Args:
        section_name: Section to check (defaults to Sigil Work)

    Returns:
        List of incomplete task dictionaries
    """
    try:
        # Use direct REST API to avoid paginator issues
        token = os.environ.get("TODOIST_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}

        # Get sections
        sections_response = requests.get(
            f"https://api.todoist.com/rest/v2/sections?project_id={EF_PROJECT_ID}",
            headers=headers,
            timeout=TIME_OUT,
        )
        sections_response.raise_for_status()
        sections = sections_response.json()

        # Find the matching section
        section_id = None
        for s in sections:
            if s["name"] == section_name:
                section_id = s["id"]
                break

        if not section_id:
            return []

        # Get tasks for this section
        tasks_response = requests.get(
            f"https://api.todoist.com/rest/v2/tasks?project_id={EF_PROJECT_ID}&section_id={section_id}",
            headers=headers,
            timeout=TIME_OUT,
        )
        tasks_response.raise_for_status()
        tasks = tasks_response.json()

        return [
            {
                "id": task["id"],
                "content": task["content"],
                "is_completed": task.get("is_completed", False),
                "url": task["url"],
                "created_at": task["created_at"],
            }
            for task in tasks
            if not task.get("is_completed", False)
        ]

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(
                f"403 Forbidden: Cannot access tasks in project {EF_PROJECT_ID}. Check API token permissions."
            )
        elif e.response.status_code == 401:
            logger.error("401 Unauthorized: Invalid API token.")
        else:
            logger.error(f"HTTP error listing tasks: {e}")
        return []
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return []


@mcp.tool()
def create_todoist_task_for_human_earthly_frame(
    content: str,
    project_id: str = EF_PROJECT_ID,
    section_id: Optional[str] = None,
    description: Optional[str] = None,
    priority: int = 1,
) -> Dict[str, Any]:
    """
    Generic task creation function for The Earthly Frames project.

    Args:
        content: Task title/content
        project_id: Todoist project ID (defaults to Earthly Frames)
        section_id: Optional section ID to place task in
        description: Optional longer description
        priority: Priority level (1=P4, 2=P3, 3=P2, 4=P1)

    Returns:
        Task dictionary with id, content, url
    """
    try:
        token = os.environ.get("TODOIST_API_TOKEN")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        task_data = {
            "content": content,
            "project_id": project_id,
            "priority": priority,
        }

        if section_id:
            task_data["section_id"] = section_id
        if description:
            task_data["description"] = description

        response = requests.post(
            "https://api.todoist.com/rest/v2/tasks",
            headers=headers,
            json=task_data,
            timeout=TIME_OUT,
        )
        response.raise_for_status()
        task = response.json()

        return {
            "id": task["id"],
            "content": task["content"],
            "url": task["url"],
            "project_id": task["project_id"],
            "section_id": task.get("section_id"),
            "created_at": task["created_at"],
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(
                f"403 Forbidden: Cannot create task in project {project_id}. Check API token permissions for project access and task creation."
            )
        elif e.response.status_code == 401:
            logger.error("401 Unauthorized: Invalid API token.")
        else:
            logger.error(f"HTTP error creating task: {e}")
        logger.error(f"Error creating task: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise


def diagnose_todoist():
    token = os.getenv("TODOIST_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    # Test basic connection
    try:
        response = requests.get(
            "https://api.todoist.com/rest/v2/projects", headers=headers, timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json() if response.ok else response.text}")
        return response.ok
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        return False
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False


# Compatibility shim: provide get_api_client() returning a small client
# that exposes the minimal methods older code expects (get_task, get_sections,
# get_tasks, add_task, add_section). Internally it uses the REST API already
# used throughout this module.

_api_client: Optional["TodoistCompat"] = None

# Expose a TodoistAPI symbol for tests/legacy code to patch. If the real
# SDK is available it can be assigned here; tests will patch it during unit tests.
TodoistAPI: Optional[Any] = None


class _AttrObj:
    """Simple object to expose dict keys as attributes."""

    def __init__(self, data: Dict[str, Any]):
        for k, v in data.items():
            setattr(self, k, v)

    def __repr__(self) -> str:  # pragma: no cover - small helper
        return f"_AttrObj({getattr(self, 'id', None)})"


class TodoistCompat:
    """Minimal compatibility wrapper around Todoist REST API.

    Methods return simple objects with attributes (like the old Pydantic models)
    so existing code that does `task.id` or `task.is_completed` continues to work.
    """

    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}

    def get_task(self, task_id: str) -> _AttrObj:
        r = requests.get(
            f"https://api.todoist.com/rest/v2/tasks/{task_id}",
            headers=self.headers,
            timeout=TIME_OUT,
        )
        r.raise_for_status()
        return _AttrObj(r.json())

    def get_sections(self, project_id: str):
        r = requests.get(
            f"https://api.todoist.com/rest/v2/sections?project_id={project_id}",
            headers=self.headers,
            timeout=TIME_OUT,
        )
        r.raise_for_status()
        return [_AttrObj(s) for s in r.json()]

    def get_tasks(
        self, project_id: Optional[str] = None, section_id: Optional[str] = None
    ):
        url = "https://api.todoist.com/rest/v2/tasks"
        params = []
        if project_id:
            params.append(f"project_id={project_id}")
        if section_id:
            params.append(f"section_id={section_id}")
        if params:
            url = url + "?" + "&".join(params)
        r = requests.get(url, headers=self.headers, timeout=TIME_OUT)
        r.raise_for_status()
        return [_AttrObj(t) for t in r.json()]

    def add_task(
        self,
        content: str,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        section_id: Optional[str] = None,
        priority: int = 1,
    ) -> _AttrObj:
        payload: Dict[str, Any] = {"content": content, "priority": priority}
        if project_id:
            payload["project_id"] = project_id
        if section_id:
            payload["section_id"] = section_id
        if description:
            payload["description"] = description
        r = requests.post(
            "https://api.todoist.com/rest/v2/tasks",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=TIME_OUT,
        )
        r.raise_for_status()
        return _AttrObj(r.json())

    def add_section(self, name: str, project_id: str) -> _AttrObj:
        payload = {"name": name, "project_id": project_id}
        r = requests.post(
            "https://api.todoist.com/rest/v2/sections",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=TIME_OUT,
        )
        r.raise_for_status()
        return _AttrObj(r.json())


def get_api_client() -> "TodoistCompat":
    """Return a singleton compatibility client for older code that imports get_api_client().

    If a `TodoistAPI` symbol is provided at module level (tests may patch this),
    it will be used to construct the client and its return value will be stored
    in the singleton. Otherwise we use the internal `TodoistCompat` wrapper.
    """
    global _api_client
    if _api_client is None:
        token = os.environ.get("TODOIST_API_TOKEN")
        if not token:
            raise ValueError("TODOIST_API_TOKEN not found in environment")
        # If a real TodoistAPI class is present (or patched in tests), use it.
        if TodoistAPI is not None:
            _api_client = TodoistAPI(token)
        else:
            _api_client = TodoistCompat(token)
    return _api_client


if __name__ == "__main__":
    mcp.run(transport="stdio")
