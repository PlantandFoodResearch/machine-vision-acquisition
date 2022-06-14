import click
import json
import logging
import cv2
import threading
from typing import Optional, Dict, List
import atexit
import time
from pathlib import Path
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper, get_camera_by_serial
from machine_vision_acquisition_python.converter.processing import resize_with_aspect_ratio, cvt_tonemap_image
from machine_vision_acquisition_python.models import Config, GenICamParam
from machine_vision_acquisition_python.utils import enable_ptp_sync, disable_ptp_sync

# temp
from flask import Flask, render_template
from flask.wrappers import Response
import os

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
    log.info(f"Opening {len(config.cameras)} cameras")
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


def liveview_web(camera: CameraHelper):
    """Attempt to live stream images to webpage"""
    web_path = (Path.cwd() / "web").resolve()
    template_path = web_path / "templates"
    if not template_path.exists():
        raise FileNotFoundError(f"Could not find template folder")
    app = Flask(__name__, static_folder=web_path, template_folder=str(template_path))

    def gen_frames():  
        while True:
            if camera.cached_image is None:
                time.sleep(0.5)
                continue
            else:
                with camera.lock:
                    img = camera.cached_image.copy()
                    camera.cached_image = None
                img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
                img = resize_with_aspect_ratio(img, width=480)
                img = cvt_tonemap_image(img)
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                log.info(f"sending web frame {len(buffer)/1024} kb")
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    @app.route('/')
    def index():
        return render_template("index.html")

    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    log.info(f"starting webserver")
    try:
        app.run(host="0.0.0.0", debug=False)
    finally:
        log.info(f"stopping webserver")



def main(config: Config):
    cameras = open_cameras(config)
    # Set all camera properties
    set_camera_params(config=config, cameras=cameras)

    if config.ptp_sync:
        enable_ptp_sync(cameras)
    elif config.ptp_sync == False:
        disable_ptp_sync(cameras)
    ts = []
    shutdown = threading.Event()
    atexit.register(shutdown.set)
    for camera in cameras:
        ts.append(time.perf_counter_ns())
        t = threading.Thread(target=camera.run_process_buffer, args=(shutdown,))
        t.start()
        camera.start_capturing()
    try:
        time.sleep(5.0)
        liveview_web(cameras[1])
        while True:
            res = cv2.waitKey(10)
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