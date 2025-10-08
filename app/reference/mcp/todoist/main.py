import logging
import os

from todoist_api_python.api import TodoistAPI
from todoist_api_python.models import Section, Task
from dotenv import load_dotenv
from typing import Any, List, Optional, Dict
from mcp.server.fastmcp import FastMCP

USER_AGENT = "earthly_frames_todoist/1.0"
TIME_OUT = 30.0
EF_PROJECT_ID = "6CrfWqXrxppjhqMJ"

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Initialize API client once at module level
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
        project_id: Todoist project ID (defaults to Earthly Frames project)

    Returns:
        List of section dictionaries with id, name, project_id, order
    """
    try:
        api = get_api_client()
        sections: List[Section] = list(api.get_sections(project_id=project_id))  # type: ignore[arg-type]

        # Convert Section objects to dictionaries
        result: List[Dict[str, Any]] = []
        for section in sections:
            result.append({
                "id": section.id,
                "name": section.name,
                "project_id": section.project_id,
                "order": section.order
            })
        return result
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
    try:
        api = get_api_client()

        # Find or create the section
        sections: List[Section] = list(api.get_sections(project_id=EF_PROJECT_ID))  # type: ignore[arg-type]
        section: Optional[Section] = None
        for s in sections:
            if s.name == section_name:
                section = s
                break

        if not section:
            # Create section if it doesn't exist
            section = api.add_section(name=section_name, project_id=EF_PROJECT_ID)

        # Format task content
        task_content = f"🜏 Charge Sigil for '{song_title}'"
        task_description = f"""
**Sigil Glyph:**
{sigil_description}

**Charging Instructions:**
{charging_instructions}

**Song:** {song_title}

Mark this task complete after the sigil has been charged and released.
"""

        # Create the task
        task: Task = api.add_task(
            content=task_content,
            description=task_description,
            project_id=EF_PROJECT_ID,
            section_id=section.id,
            priority=3  # P2 priority (Black Agent's work is urgent)
        )

        # Convert Task object to dictionary
        return {
            "id": task.id,
            "content": task.content,
            "url": task.url,
            "project_id": task.project_id,
            "section_id": task.section_id,
            "created_at": task.created_at
        }

    except Exception as e:
        logging.error(f"Error creating sigil charging task: {e}")
        raise


@mcp.tool()
def create_evp_analysis_task(
        transcript: str,
        audio_file_path: str,
        song_title: str,
        section_name: str = "Black Agent - EVP Analysis"
) -> Dict[str, Any]:
    """
    Create a Todoist task for human to analyze EVP transcript.

    Args:
        transcript: The hallucinated transcript from EVP generation
        audio_file_path: Path to the noise-blended audio file
        song_title: Title of the song this EVP is for
        section_name: Name of the Todoist section

    Returns:
        Task dictionary with id, content, url
    """
    try:
        api = get_api_client()

        # Find or create section
        sections: List[Section] = list(api.get_sections(project_id=EF_PROJECT_ID))  # type: ignore[arg-type]
        section: Optional[Section] = None
        for s in sections:
            if s.name == section_name:
                section = s
                break

        if not section:
            section = api.add_section(name=section_name, project_id=EF_PROJECT_ID)

        task_content = f"👻 Review EVP for '{song_title}'"
        task_description = f"""
**EVP Transcript:**
{transcript}

**Audio File:**
{audio_file_path}

Listen to the blended audio and confirm if the transcript captures meaningful spirit messages.
Adjust the counter-proposal based on what you hear.

Mark complete after review.
"""

        task: Task = api.add_task(
            content=task_content,
            description=task_description,
            project_id=EF_PROJECT_ID,
            section_id=section.id,
            priority=2  # P3 priority (analysis, not ritual)
        )

        return {
            "id": task.id,
            "content": task.content,
            "url": task.url,
            "project_id": task.project_id,
            "section_id": task.section_id
        }

    except Exception as e:
        logging.error(f"Error creating EVP analysis task: {e}")
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
    try:
        api = get_api_client()

        # Get section ID
        sections: List[Section] = list(api.get_sections(project_id=EF_PROJECT_ID))  # type: ignore[arg-type]
        section: Optional[Section] = None
        for s in sections:
            if s.name == section_name:
                section = s
                break

        if not section:
            return []

        # Get all tasks in project
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

    except Exception as e:
        logging.error(f"Error listing tasks: {e}")
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
    Generic task creation function for Earthly Frames project.

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
    except Exception as e:
        logging.error(f"Error creating task: {e}")
        raise