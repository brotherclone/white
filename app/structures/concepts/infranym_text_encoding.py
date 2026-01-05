from abc import ABC

from pydantic import BaseModel, Field

from app.structures.enums.infranym_method import InfranymMethod


class InfranymTextEncoding(BaseModel, ABC):
    """Base encoding data for text infranyms"""

    method: InfranymMethod
    secret_word: str = Field(..., description="The hidden word/phrase")
    surface_text: str = Field(..., description="The visible text (full lyrics/poem)")
