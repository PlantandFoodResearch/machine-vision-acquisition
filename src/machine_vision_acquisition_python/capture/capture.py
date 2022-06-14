import click
import json
import logging
import cv2
from typing import Optional, Dict, List
import time
from pathlib import Path
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper, get_camera_by_serial
from machine_vision_acquisition_python.converter.processing import resize_with_aspect_ratio
from machine_vision_acquisition_python.models import Config, GenICamParam
from machine_vision_acquisition_python.utils import enable_ptp_sync, disable_ptp_sync
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


def set_camera_params(config: Config, cameras: List[CameraHelper]) -> None:
    for camera in cameras:
        serial = camera.camera.get_device_serial_number()
        camera_config = config.get_camera_config_by_serial(serial)
        params: List[GenICamParam] = camera_config.params or []
        # Set shared params
        if config.shared_params:
            params += config.shared_params
        # Set all params
        # todo: consider use of set_features_from_string to bulk set
        for param in params:
            camera.set_parameter(param)



def temp_display_latest(cameras: List[CameraHelper]):
    for camera in cameras:
        cv2.imshow(camera.name, resize_with_aspect_ratio(camera.cached_image, width=480))


def main(config: Config):
    cameras = open_cameras(config)
    # Set all camera properties
    set_camera_params(config=config, cameras=cameras)

    if config.ptp_sync:
        enable_ptp_sync(cameras)
    elif config.ptp_sync == False:
        disable_ptp_sync(cameras)
    ts = []
    for camera in cameras:
        ts.append(time.perf_counter_ns())
        camera.start_capturing()
        # camera.camera.software_trigger()
    # while True:
    #     cv2.waitKey(500)
    #     for camera in cameras:
    #         camera.unpack_last_buffer()
    #     temp_display_latest(cameras)
    # # log.info(f"ts: {ts}")
    pass