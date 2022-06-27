from pathlib import Path
from typing import List, Optional, Union, Literal
from pydantic import (
    BaseModel,
    BaseSettings,
    Field,
    validator,
    root_validator,
    DirectoryPath,
)


class GenICamParam(BaseModel):
    """Generic GenICam parameter"""

    name: str = Field(description="GenICam parameter name, case sensitive")
    value: Union[float, int, str, bool] = Field()
    # is validated to:  Optional[Union[Type[bool],Type[str],Type[int],Type[float]]]
    val_type: Optional[Literal["str", "int", "float", "bool"]] = Field(
        description="Force type when setting, otherwise a guess will be attempted"
    )

    @root_validator()
    def ensure_type(cls, values):
        if values["val_type"]:
            type_str = values["val_type"]
            if type_str == "bool":
                val_type = bool
            elif type_str == "str":
                val_type = str
            elif type_str == "int":
                val_type = int
            elif type_str == "float":
                val_type = float
            else:
                raise ValueError(f"Unsupported type {type_str}")
            values["value"] = val_type(values["value"])
            values["val_type"] = val_type
        return values


class Camera(BaseModel):
    """Aravis camera configuration"""

    serial: str
    params: Optional[List[GenICamParam]] = Field(
        description="GenICam parameters to this camera. Note: the order will attempt to be respected"
    )


class Config(BaseSettings):
    """Configuration for machine_vision_acquisition_python.capture CLI program"""

    version: str
    cameras: List[Camera] = Field(description="Cameras to open")
    ptp_sync: Optional[bool] = Field(
        description="Attempt to enable GigE PTP sync between cameras"
    )
    shared_params: Optional[List[GenICamParam]] = Field(
        description="GenICam parameters to apply to all opened cameras. Note: the order will attempt to be respected"
    )
    output_directory: DirectoryPath = Field(
        description="Path to output folder to use as root. Will be created (including parents) if required"
    )

    # This enables auto load from environment variables
    class Config:
        env_prefix = "pycapture_"
        case_sensitive = False
        smart_union = True  # attempt to preserve JSON type for Unions

    @validator("cameras")
    def cameras_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError("cameras list must not be empty")
        return v

    def get_camera_config_by_serial(self, serial: str) -> Camera:
        for camera_config in self.cameras:
            if camera_config.serial == serial:
                return camera_config
        raise ValueError(f"Camera with serial {serial} was not found in config")
