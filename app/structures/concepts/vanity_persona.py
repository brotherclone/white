from pydantic import BaseModel


class VanityPersona(BaseModel):

    first_name: str
    last_name: str
