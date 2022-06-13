import click
import json
import logging
from typing import Optional, Dict, List
from pathlib import Path
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper, get_camera_by_serial
from machine_vision_acquisition_python.models import Config
log = logging.getLogger(__name__)

@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    help="Path to JSON configuration file for capture",
    required=True,
    type=click.types.Path(file_okay=True, exists=True, dir_okay=False, readable=True, path_type=Path)
)
def cli(config_path: Path):
    config: Config = Config(**json.loads(config_path.read_text()))
    log.debug(f"Opening {len(config.cameras)} cameras")
    main(config)


def open_cameras(config: Config) -> List[CameraHelper]:
    cameras = []
    for camera in config.cameras:
        helper = get_camera_by_serial(camera.serial)
        # do validity checks
        cameras.append(helper)
    if len(cameras) == 0:
        raise ValueError("Was unable to open any cameras")
    return cameras


def main(config: Config):
    cameras = open_cameras(config)
    pass