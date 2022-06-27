import chronoptics.tof as tof
import cv2
import typing
import logging
import numpy as np
import time
import click
from timeit import default_timer as timer
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper
from machine_vision_acquisition_python.interfaces.chronoptics import ToFCameraHelper
from machine_vision_acquisition_python.process.processing import (
    resize_with_aspect_ratio,
)


log = logging.getLogger(__name__)


def getFrame(frames: typing.List[tof.Data], frame_type: tof.FrameType):
    for frame in frames:
        if frame.frameType() == frame_type:
            return frame
    return None


def get_first_valid_kea_camera() -> ToFCameraHelper:
    interface = tof.GigeInterface()
    msgs = interface.discover()
    camera = None
    for msg in msgs:
        try:
            camera = ToFCameraHelper(msg.serial())
            return camera
        except RuntimeError as _:
            continue
    raise ValueError("Could not find a valid KeaCamera")


@click.command()
@click.option(
    "--serial", help="KeaCamera serial to try", default=None, required=False, type=str
)
def cli(serial: str):
    """
    Simple Kea python viewer. Shows RGB, Intensity, and Radial Depth. *Requires a display*.
    Hotkeys:\n
    * 'q' to exit\n
    """
    if not serial:
        camera = get_first_valid_kea_camera()
    else:
        camera = ToFCameraHelper(serial)
    cv2.namedWindow(f"{camera.name}-RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow(f"{camera.name}-Intensity", cv2.WINDOW_NORMAL)
    cv2.namedWindow(f"{camera.name}-Radial", cv2.WINDOW_NORMAL)

    log.setLevel(logging.INFO)  # Reduce spam during high FPS

    try:
        while True:
            ch = chr(cv2.waitKey(10) & 0xFF)
            if ch == "q":
                break
            # camera.get_cache_all_rgb_intensity_radial()
            cv2.imshow(f"{camera.name}-RGB", camera.get_single_image())
            cv2.imshow(f"{camera.name}-Intensity", camera.get_normalised_intensity())
            cv2.imshow(f"{camera.name}-Radial", camera.cached_image_radial)
            time.sleep(0.1)
    finally:
        cv2.destroyAllWindows()
        camera.camera.stop()


if __name__ == "__main__":
    cli()
