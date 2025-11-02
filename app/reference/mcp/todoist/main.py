import logging
import os


from todoist_api_python.api import TodoistAPI
from todoist_api_python.models import Section, Task
from requests.exceptions import HTTPError
from dotenv import load_dotenv
from typing import Any, List, Optional, Dict
from mcp.server.fastmcp import FastMCP

from app.util.string_utils import resolve_name

USER_AGENT = "earthly_frames_todoist/1.0"
TIME_OUT = 30.0
EF_PROJECT_ID = "6CrfWqXrxppjhqMJ"

logging.basicConfig(level=logging.INFO)

load_dotenv()

_api_client: Optional[TodoistAPI] = None

def get_api_client() -> TodoistAPI:
    """Get or create TodoistAPI client singleton"""
    global _api_client
    if _api_client is None:
        api_token = os.environ.get('TODOIST_API_TOKEN')
        if not api_token:
            raise ValueError("TODOIST_API_TOKEN not found in environment")
        _api_client = TodoistAPI(api_token)
    return _api_client

mcp = FastMCP("earthly_frames_todoist")

@mcp.tool()
def get_earthly_frames_project_sections(project_id: str = EF_PROJECT_ID) -> List[Dict[str, Any]]:
    """
    Get all sections for the Earthly Frames Todoist project.

    Args:
        project_id: Todoist project ID (defaults to The Earthly Frames project)

    Returns:
        List of section dictionaries with id, name, project_id, order
    """
    try:
        api = get_api_client()
        sections: List[Section] = list(api.get_sections(project_id=project_id))
        result: List[Dict[str, Any]] = []
        for section in sections:
            result.append({
                "id": section.id,
                "name": resolve_name(section),
                "project_id": section.project_id,
                "order": section.order
            })
        return result
    except HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 403:
            logging.error(f"403 Forbidden: Access denied to project {project_id}. Check API token permissions.")
        elif hasattr(e, 'response') and e.response.status_code == 401:
            logging.error(f"401 Unauthorized: Invalid API token for project {project_id}.")
        else:
            logging.error(f"HTTP error fetching sections for project {project_id}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching sections for project {project_id}: {e}")
        return []


@mcp.tool()
def create_sigil_charging_task(
        sigil_description: str,
        charging_instructions: str,
        song_title: str,
        section_name: str = "Black Agent - Sigil Work"
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
    """
    sections: List[Section] = []
    try:
        api = get_api_client()
        sections = list(api.get_sections(project_id=EF_PROJECT_ID))  # type: ignore[arg-type]
        section: Optional[Section] = None
        for s in sections:
            name = resolve_name(s)
            if name == section_name:
                section = s
                break
        if not section:
            section = api.add_section(name=section_name, project_id=EF_PROJECT_ID)
        task_content = f"🜏 Charge Sigil for '{song_title}'"
        task_description = f"""
                            **Sigil Glyph:**
                            {sigil_description}

                            **Charging Instructions:**
                            {charging_instructions}

                            **Song:** {song_title}

                            Mark this task complete after the sigil has been charged and released.
                            """
        task: Task = api.add_task(
            content=task_content,
            description=task_description,
            project_id=EF_PROJECT_ID,
            section_id=section.id,
            priority=3
        )
        return {
            "id": task.id,
            "content": task.content,
            "url": task.url,
            "project_id": task.project_id,
            "section_id": task.section_id,
            "created_at": task.created_at
        }

    except HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 403:
            logging.error(f"403 Forbidden: Cannot create task in project {EF_PROJECT_ID}. Check API token permissions for project access and task creation.")
            logging.exception("Failed creating sigil charging task; sections=%r", sections)
        elif hasattr(e, 'response') and e.response.status_code == 401:
            logging.error(f"401 Unauthorized: Invalid API token.")
            logging.exception("Failed creating sigil charging task; sections=%r", sections)
        else:
            logging.error(f"HTTP error creating sigil charging task: {e}")
            logging.exception("Failed creating sigil charging task; sections=%r", sections)
        raise
    except Exception:
        logging.exception("Failed creating sigil charging task; sections=%r", sections)
        raise

@mcp.tool()
def list_pending_black_agent_tasks(section_name: str = "Black Agent - Sigil Work") -> List[Dict[str, Any]]:
    """
    List all incomplete tasks in Black Agent sections.

    Args:
        section_name: Section to check (defaults to Sigil Work)

    Returns:
        List of incomplete task dictionaries
    """
    sections: List[Section] = []
    try:
        api = get_api_client()
        sections = list(api.get_sections(project_id=EF_PROJECT_ID))  # type: ignore[arg-type]
        section: Optional[Section] = None
        for s in sections:
            if resolve_name(s) == section_name:
                section = s
                break

        if not section:
            return []
        tasks: List[Task] = list(api.get_tasks(project_id=EF_PROJECT_ID, section_id=section.id))  # type: ignore[arg-type]
        return [
            {
                "id": task.id,
                "content": task.content,
                "is_completed": task.is_completed,
                "url": task.url,
                "created_at": task.created_at
            }
            for task in tasks
            if not task.is_completed
        ]

    except HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 403:
            logging.error(f"403 Forbidden: Cannot access tasks in project {EF_PROJECT_ID}. Check API token permissions.")
        elif hasattr(e, 'response') and e.response.status_code == 401:
            logging.error(f"401 Unauthorized: Invalid API token.")
        else:
            logging.error(f"HTTP error listing tasks: {e}")
        logging.exception(f"Error listing tasks; sections=%r", sections)
        return []
    except ValueError as e:
        logging.exception(f"Error listing tasks;{e}; sections=%r", sections)
        return []


@mcp.tool()
def create_todoist_task_for_human_earthly_frame(
        content: str,
        project_id: str = EF_PROJECT_ID,
        section_id: Optional[str] = None,
        description: Optional[str] = None,
        priority: int = 1
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
        api = get_api_client()
        task: Task = api.add_task(
            content=content,
            description=description,
            project_id=project_id,
            section_id=section_id,
            priority=priority
        )
        return {
            "id": task.id,
            "content": task.content,
            "url": task.url,
            "project_id": task.project_id,
            "section_id": task.section_id,
            "created_at": task.created_at
        }
    except HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 403:
            logging.error(f"403 Forbidden: Cannot create task in project {project_id}. Check API token permissions for project access and task creation.")
        elif hasattr(e, 'response') and e.response.status_code == 401:
            logging.error(f"401 Unauthorized: Invalid API token.")
        else:
            logging.error(f"HTTP error creating task: {e}")
        logging.error(f"Error creating task: {e}")
        raise
    except Exception as e:
        logging.error(f"Error creating task: {e}")
        raise

if __name__ == "__main__":
    mcp.run()