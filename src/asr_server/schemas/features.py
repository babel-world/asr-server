from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class FeaturesModelStateResponseBody(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    loaded: bool = False
    message: str = ""
