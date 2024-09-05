from pydantic import BaseModel, ConfigDict

class TextRequest(BaseModel):
    Title: str
    Body: str
    model_config = ConfigDict(arbitrary_types_allowed=True)