from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class TranscribeRequestBody(BaseModel):
    """Multipart form-data: field `file` (.wav audio only)."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class TranscribeResponseBody(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    transcribed_text: str = ""
    language: str = ""
    language_probability: float = 0.0


class TranscribeModelStateResponseBody(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    loaded: bool = False
    message: str = ""
