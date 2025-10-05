import logging
import os

from todoist_api_python.api import TodoistAPI
from dotenv import load_dotenv
from typing import Any
from mcp.server.fastmcp import FastMCP

USER_AGENT = "earthly_frames_todoist/1.0"
TIME_OUT = 30.0
EF_PROJECT_ID="6CrfWqXrxppjhqMJ"

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("earthly_frames_todoist")

@mcp.tool()
async def todoist_earthly_frames_service() -> Any:
    load_dotenv()
    client_id = os.environ['TODOIST_API_TOKEN']
    api = TodoistAPI(client_id)
    return api

@mcp.tool()
async def get_earthly_frames_todoist_project_sections(api: TodoistAPI, project_id: str):
    try:
        sections = api.get_sections(project_id=project_id)
        return sections
    except Exception as e:
        logging.error(f"Error fetching sections for project {project_id}: {e}")
        return []

@mcp.tool()
async def create_todoist_task_for_human_earthly_frame(api: TodoistAPI, content: str, project_id: str, section_id: str):
    try:
        task = api.add_task(content=content, project_id=project_id, section_id=section_id)
        return task
    except Exception as e:
        logging.error(f"Error creating task: {e}")
        return None

mcp.tools = [
    todoist_earthly_frames_service,
    get_earthly_frames_todoist_project_sections,
    create_todoist_task_for_human_earthly_frame
]

