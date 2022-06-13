from typing import List, Optional
from pydantic import BaseModel, BaseSettings, Field, validator


class Camera(BaseModel):
    serial: str

class Config(BaseSettings):
    version: str
    cameras: List[Camera]
    ptp_sync: Optional[bool]

    # This enables auto load from environment variables
    class Config:
        env_prefix = 'pycapture_'
        case_sensitive = False

    @validator('cameras')
    def cameras_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('cameras list must not be empty')
        return v