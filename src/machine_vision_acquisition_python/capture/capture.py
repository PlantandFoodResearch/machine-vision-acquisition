import click
import json
import logging
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
import atexit
import time
import functools
import numpy as np
from pathlib import Path
from machine_vision_acquisition_python.interfaces.aravis import (
    CameraHelper,
    get_camera_by_serial,
)
from machine_vision_acquisition_python.process.processing import (
    resize_with_aspect_ratio,
    cvt_tonemap_image,
)
from machine_vision_acquisition_python.models import Config, GenICamParam
from machine_vision_acquisition_python.utils import enable_ptp_sync, disable_ptp_sync
from machine_vision_acquisition_python.capture.keyboard import register_callback

# Temporary helpers
from machine_vision_acquisition_python.capture.misc import *

# temp
from flask import Flask, render_template
from flask.wrappers import Response
import os

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    help="Path to JSON configuration file for capture",
    required=True,
    type=click.types.Path(
        file_okay=True, exists=True, dir_okay=False, readable=True, path_type=Path
    ),
)
@click.option(
    "--output",
    "-o",
    "out_dir",
    help="Path to output folder to use as root. Will be created (including parents) if required",
    required=False,
    type=click.types.Path(
        file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.option(
    "--webviewer",
    "-w",
    help="Run a webserver that hosts the output streams (experimental!). The port it is served on can be controlled with the environment variable 'HTTP_PORT'.",
    is_flag=True,
    default=False,
)
def cli(config_path: Path, out_dir: Optional[Path], webviewer: bool):
    """
    Basic camera capturing from a config file. Once loaded, will attempt to open and set all parameters, then begin acquisition.
    Can host basic webpage to preview the live camera output.

    Hotkeys while running:\n
    - s: Save current frame to disk (uses `output`)\n
    """
    if out_dir is None:
        out_dir = Path.cwd() / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    json_config: dict = json.loads(config_path.read_text())
    out_dir = Path(json_config.setdefault("output_directory", str(out_dir)))
    config: Config = Config(**json_config)
    log.info(f"Opening {len(config.cameras)} cameras with {out_dir}")
    main(config, webviewer)


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


def save_current_frame(
    camera: CameraHelper, out_dir: Optional[Path] = None, debayer: bool = True, tonemap: bool = True, image_index: Optional[int] = None
):
    if out_dir is None:
        out_dir = Path.cwd().resolve() / "tmp" / camera.short_name
    with camera.lock:
        if camera.cached_image is None:
            log.warning(f"cannot save image for {camera.name}, none cached")
            return
        image = camera.cached_image.copy()
        image_time = camera.cached_image_time
    if image is None or image_time is None:
        log.warning(f"cannot save image for {camera.name}, none cached")
        return
    if debayer:
        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    if tonemap:
        image = cvt_tonemap_image(image)
    # Will result in  YYYY-MM-DDTHH-mm-ss-[ms*3] e.g. 2022-06-20T00-22-44-209
    pathsafe_time_str = (
        np.datetime_as_string(image_time, unit="ms").replace(":", "-").replace(".", "-")
    )
    # Will give names like: "Grasshopper3-GS3-U3-23S6C-15122686-2022-06-20T00-22-44-209.png"
    
    # Can be overridden to be a simpe index (for calibration)
    img_path = out_dir / f"{camera.short_name}-{pathsafe_time_str}.png"
    if image_index:
        img_path = out_dir / f"{image_index}-{camera.short_name}-{pathsafe_time_str}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fpnge
        fpnge_bytes = fpnge.fromMat(image)
        img_path.write_bytes(fpnge_bytes)
    except ImportError as _:
        log.warning(f"Must install python fpnge bindings for faster PNG saving: https://github.com/animetosho/python-fpnge.")
        if not cv2.imwrite(str(img_path), image):
            raise ValueError("Could not write PNG file")
    log.info(f"{img_path.name} saved")


def save_all_images_cb(cameras: List[CameraHelper], root_dir: Path):
    # Trigger all
    for camera in cameras:
        camera.camera.software_trigger()
    # Cache the worker pool
    if getattr(save_all_images_cb, "executor", None) is None:
        save_all_images_cb.executor = ThreadPoolExecutor(max_workers=len(cameras))
    exec: ThreadPoolExecutor = save_all_images_cb.executor
    for camera in cameras:
        out_dir = Path.cwd().resolve() / "tmp" / camera.short_name
        job = functools.partial(save_current_frame, camera, out_dir, debayer=True, tonemap=True, image_index=save_all_images_cb.index)
        exec.submit(job)
    save_all_images_cb.index += 1
save_all_images_cb.index = 1

def main(config: Config, webviewer):
    cameras = open_cameras(config)
    # Set all camera properties
    set_camera_params(config=config, cameras=cameras)

    shutdown = threading.Event()
    atexit.register(shutdown.set)
    soft_trigger_cameras: List[CameraHelper] = []
    external_trigger_cameras: List[CameraHelper] = []
    for camera in cameras:
        t = threading.Thread(target=camera.run_process_buffer, args=(shutdown,))
        t.start()
        if camera.camera.get_trigger_source() == "Software":
            soft_trigger_cameras.append(camera)
        else:
            external_trigger_cameras.append(camera)

    if config.ptp_sync:
        enable_ptp_sync(cameras)
    elif config.ptp_sync == False:
        disable_ptp_sync(cameras)

    ts = []

    # Stop all
    for camera in cameras:
        camera.camera.stop_acquisition()
    time.sleep(0.5)
    # Start software triggered
    for camera in soft_trigger_cameras:
        camera.start_capturing()
        camera.camera.software_trigger()
    time.sleep(0.5)
    # Start external triggered
    for camera in external_trigger_cameras:
        camera.start_capturing()
    time.sleep(0.5)
    # for camera in cameras:
    #     camera.camera.stop_acquisition()
    #     ts.append(time.perf_counter_ns())
    #     t = threading.Thread(target=camera.run_process_buffer, args=(shutdown,))
    #     t.start()
    #     camera.start_capturing()
    try:
        cb = functools.partial(save_all_images_cb, cameras, config.output_directory)
        register_callback("s", cb)
        if webviewer:
            # blocks forever
            liveview_web(cameras)
        else:
            # sleep forever
            log.info(f"Setup done, press 's' to capture an image or CTRL-C to exit")
            while True:
                time.sleep(1.0)
        # test_print_all(cameras)
        # while True:
        # time.sleep(5.0)
        # cam: CameraHelper = soft_trigger_cameras[0]
        # cam.camera.software_trigger()
        # res = cv2.waitKey(100)
        # if res <= 0:
        #     continue
        # ch = chr(res)
        # if ch == "t":
        #     cam.camera.software_trigger()
    except KeyboardInterrupt as _:
        pass
    finally:
        shutdown.set()
        # camera.camera.software_trigger()
    # while True:
    #     cv2.waitKey(500)
    #     for camera in cameras:
    #         camera.unpack_last_buffer()
    #     temp_display_latest(cameras)
    # # log.info(f"ts: {ts}")
    pass
