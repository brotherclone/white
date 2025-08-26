from pydantic import BaseModel


class BaseManifestExtractor(BaseModel):

    def __init__(self, **data):
        super().__init__(**data)


