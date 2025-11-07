import os

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

load_dotenv()


class AgentSettings(BaseModel):

    anthropic_api_key: SecretStr = SecretStr(os.getenv("ANTHROPIC_API_KEY") or "")
    anthropic_model_name: str = "claude-sonnet-4-20250514"
    anthropic_sub_model_name: str = "claude-sonnet-4-20250514"
    work_product_path: str = os.getenv("AGENT_WORK_PRODUCT_PATH") or "/tmp/agent_work"
    temperature: float = 0.7
    max_retries: int = 3
    timeout: int = 120
    stop: list[str] = []
