from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

    @field_validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query is too long (max 1000 characters)")
        return v.strip()


class ChatResponse(BaseModel):
    response: str
