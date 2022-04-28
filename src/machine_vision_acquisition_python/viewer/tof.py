import chronoptics.tof as tof
import cv2
import typing
import logging
import numpy as np
import time
from timeit import default_timer as timer
from machine_vision_acquisition_python.viewer.cli import CameraHelper, resize_with_aspect_ratio, DISPLAY_SIZE_WIDTH

log = logging.getLogger(__name__)


def getFrame(frames: typing.List[tof.Data], frame_type: tof.FrameType):
    for frame in frames:
        if frame.frameType() == frame_type:
            return frame
    return None


class ToFCameraHelper(CameraHelper):
    def __init__(self, name: typing.Optional[str]) -> None:
        if not name:
            raise ValueError("Name is not optional for Chronoptics cameras")
        try:
            self.camera: tof.KeaCamera = tof.KeaCamera(tof.ProcessingConfig(), name)
            # TODO: Config and process config need tuning. Coppied from Chronoptics example "rgb_depth_stream.py"
            config = self.camera.getCameraConfig()
            config.reset()
            config.setModulationFrequency(0, 80.0)
            config.setPhaseShifts(0, [0.0, 0.25, 0.375, 0.625])
            config.setIntegrationTime(0, [300, 300, 300, 300])
            config.setDutyCycle(0, 0.4)
            config.setFlip(0, False)  # These appear to not work?! (At least on the RGB stream)
            config.setMirror(0, False)  # These appear to not work?! (At least on the RGB stream)
            # config.addFrame()
            # config.setModulationFrequency(1, 100.0)
            # config.setPhaseShifts(1, [0.0, 0.25, 0.375, 0.625])
            # config.setIntegrationTime(1, [300, 300, 300, 300])
            # config.setDutyCycle(1, 0.4)
            # config.setFlip(1, True)
            # config.setMirror(1, True)
            self.camera.setCameraConfig(config)
            # Processing config
            # proc_config = tof.ProcessingConfig()
            # proc_config.setTemporalEnabled(True)
            # proc_config.setMedianEnabled(True)
            # proc_config.setPhaseUnwrappingEnabled(True)
            # proc_config.setPhaseUnwrappingMaxOffset(np.pi / 2)
            # self.camera.setProcessConfig(proc_config)
            self.name = f"Chronoptics-{name}"

            # Start Grey / Z streaming.
            # types = [tof.FrameType.BGR, tof.FrameType.RADIAL, tof.FrameType.INTENSITY]
            types = [tof.FrameType.BGR]
            tof.selectStreams(
                self.camera, types  # type: ignore
            )
            self.camera.start()
        except Exception as exc:
            log.exception("Could not open camera")
        log.info(f"Opened {self.name}")

    def get_single_image(self):
        if self.camera.isStreaming():
            start = timer()
            frames = self.camera.getFrames()
            bgr_frame = getFrame(frames, tof.FrameType.BGR)
            if bgr_frame is None:
                raise ValueError("Could not get BGR from camera")
            # It comes out as a BGR, but displays as an RGB.
            # BUG: This is not converting the channels correctly. Maybe input isn't BGR?
            rgb_image = cv2.cvtColor(np.asarray(bgr_frame), cv2.COLOR_BGR2RGB)
            # mirroring in config isn't working?
            rgb_image = cv2.flip(rgb_image, flipCode=-1)  # flip & mirror
            self.cached_image = rgb_image  # Cache the raw image for optional saving
            self.cached_image_time = time.strftime("%Y-%m-%dT%H%M%S")
            image = resize_with_aspect_ratio(rgb_image, width=DISPLAY_SIZE_WIDTH)
            end = timer()
            log.debug(f"Acquiring image for {self.name} took: {end - start}")
            return image
        raise ValueError("Bad things")
