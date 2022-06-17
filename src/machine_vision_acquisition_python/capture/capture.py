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


def liveview_web(cameras: List[CameraHelper]):
    """Attempt to live stream images to webpage"""
    web_path = (Path.cwd() / "web").resolve()
    template_path = web_path / "templates"
    if not template_path.exists():
        raise FileNotFoundError(f"Could not find template folder")
    app = Flask(__name__, static_folder=web_path, template_folder=str(template_path))

    def gen_frames(camera: CameraHelper):  
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
                # log.info(f"sending web frame {len(buffer)/1024} kb")
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    @app.route('/')
    def index():
        return render_template("index.html")

    @app.route('/video_feed_1')
    def video_feed_1():
        return Response(gen_frames(cameras[0]), mimetype='multipart/x-mixed-replace; boundary=frame')
    @app.route('/video_feed_2')
    def video_feed_2():
        return Response(gen_frames(cameras[1]), mimetype='multipart/x-mixed-replace; boundary=frame')
    @app.route('/video_feed_3')
    def video_feed_3():
        return Response(gen_frames(cameras[2]), mimetype='multipart/x-mixed-replace; boundary=frame')

    log.info(f"starting webserver")
    try:
        app.run(host="0.0.0.0", debug=False)
    finally:
        log.info(f"stopping webserver")



def main(config: Config):
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
        # liveview_web(cameras)
        test_print_all(cameras)
        while True:
            time.sleep(5.0)
            cam: CameraHelper = soft_trigger_cameras[0]
            cam.camera.software_trigger()
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

def test_stop_all(cameras: List[CameraHelper]):
    for camera in cameras:
        camera.camera.stop_acquisition()

def test_start_all(cameras: List[CameraHelper]):
    for camera in cameras:
        camera.camera.start_acquisition()

def test_trigger_all(cameras: List[CameraHelper]):
    for camera in cameras:
        camera.camera.software_trigger()

def test_cont_all(cameras: List[CameraHelper]):
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    for camera in cameras:
        camera.camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)

def test_sing_all(cameras: List[CameraHelper]):
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    for camera in cameras:
        camera.camera.set_acquisition_mode(Aravis.AcquisitionMode.SINGLE_FRAME)

def test_print_all(cameras: List[CameraHelper]):
    for camera in cameras:
        device = camera.device
        str_out = f"""
        {camera.name}:
        AcquisitionMode: {device.get_feature('AcquisitionMode').get_value_as_string()}
        TriggerMode: {device.get_feature('TriggerMode').get_value_as_string()}
        TriggerSource: {device.get_feature('TriggerSource').get_value_as_string()}
        SingleFrameAcquisitionMode: {device.get_feature('SingleFrameAcquisitionMode').get_value_as_string()}\n
        AcquisitionStatusSelector: {device.get_feature('AcquisitionStatusSelector').get_value_as_string()}\n
        AcquisitionFrameRateEnabled: {device.get_feature('AcquisitionFrameRateEnabled').get_value_as_string()}\n
        AcquisitionStatus: {device.get_feature('AcquisitionStatus').get_value_as_string()}\n
        """
        log.info(str_out)