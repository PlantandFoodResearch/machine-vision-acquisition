import chronoptics.tof as tof
import typing
import logging
import numpy as np
import time
from machine_vision_acquisition_python.viewer.cli import CameraHelper

log = logging.getLogger(__name__)

class ToFCameraHelper(CameraHelper):
    def __init__(self, name: typing.Optional[str]) -> None:
        if not name:
            raise ValueError("Name is not optional for Chronoptics cameras")
        try:
            self.camera: tof.KeaCamera = tof.KeaCamera(tof.ProcessingConfig(), name)
            config = self.camera.getCameraConfig()
            config.reset()
            self.name = f"Chronoptics-{name}"

            # Start Grey / Z streaming.
            types = [tof.FrameType.INTENSITY, tof.FrameType.Z]
            tof.selectStreams(
                self.camera, types  # type: ignore
            )
            self.camera.start()
        except Exception as exc:
            log.exception("Could not open camera")
        log.info(f"Opened {self.name}")

    def get_single_image(self):
        if self.camera.isStreaming():
            frames = self.camera.getFrames()
            gray = np.uint8(np.squeeze(np.asarray(frames[0])))
            zframe = np.uint8((np.asarray(frames[1]) / 2700.0) * 255)
            self.cached_image = gray  # Cache the raw image for optional saving
            self.cached_image_time = time.strftime("%Y-%m-%dT%H%M%S")
            return gray
        raise ValueError("Bad things")
