# type: ignore
import click
import json
import logging
import cv2
import threading
from typing import Optional, Dict, List
import atexit
from flask import Flask, render_template
from flask.wrappers import Response
import time
from pathlib import Path
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper, get_camera_by_serial
from machine_vision_acquisition_python.process.processing import resize_with_aspect_ratio, cvt_tonemap_image
from machine_vision_acquisition_python.models import Config, GenICamParam
from machine_vision_acquisition_python.utils import enable_ptp_sync, disable_ptp_sync
import machine_vision_acquisition_python.capture.keyboard

log = logging.getLogger(__name__)

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
                time.sleep(0.1)
                continue
            else:
                with camera.lock:
                    img = camera.cached_image.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
                img = resize_with_aspect_ratio(img, width=480)
                img = cvt_tonemap_image(img)
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                # log.info(f"sending web frame {len(buffer)/1024} kb")
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                # max 1FPS
                time.sleep(1.0)
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
    @app.route('/video_feed_4')
    def video_feed_4():
        return Response(gen_frames(cameras[3]), mimetype='multipart/x-mixed-replace; boundary=frame')

    log.info(f"starting webserver")
    try:
        app.run(host="0.0.0.0", debug=False)
    finally:
        log.info(f"stopping webserver")