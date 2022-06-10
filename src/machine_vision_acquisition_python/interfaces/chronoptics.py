import chronoptics.tof as tof
import cv2
import typing
import numpy as np
import time
import logging
from timeit import default_timer as timer
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper
log = logging.getLogger(__name__)

class ToFCameraHelper(CameraHelper):
    def __init__(self, name: typing.Optional[str]) -> None:
        if not name:
            raise ValueError("Name is not optional for Chronoptics cameras")
        try:
            self.camera: tof.KeaCamera = tof.KeaCamera(tof.ProcessingConfig(), name)
            # TODO: Config and process config need tuning. Coppied from Chronoptics example "rgb_depth_stream.py"
            config = self.camera.getCameraConfig()
            config.reset()
            # config.setFrameTime(0, 100000)
            config.setModulationFrequency(0, 80.0)
            config.setPhaseShifts(0, [0.0, 0.25, 0.375, 0.625])
            config.setIntegrationTime(0, [300, 300, 300, 300])
            config.setDutyCycle(0, 0.4)
            # These do not afect the RGB stream. It must be manually flipped / mirrored
            config.setFlip(0, True)
            config.setMirror(0, True)
            config.addFrame()
            config.setModulationFrequency(1, 100.0)
            config.setPhaseShifts(1, [0.0, 0.25, 0.375, 0.625])
            config.setIntegrationTime(1, [300, 300, 300, 300])
            config.setDutyCycle(1, 0.4)
            config.setFlip(1, True)
            config.setMirror(1, True)
            self.camera.setCameraConfig(config)
            # Processing config. Coppied from Chronoptics example "rgb_depth_stream.py"
            proc_config = tof.ProcessingConfig()
            proc_config.setTemporalEnabled(True)
            proc_config.setMedianEnabled(True)
            proc_config.setPhaseUnwrappingEnabled(True)
            proc_config.setPhaseUnwrappingMaxOffset(np.pi / 2)
            self.camera.setProcessConfig(proc_config)
            self.name = f"Chronoptics-{name}"

            # Start Grey / Z streaming.
            types = [tof.FrameType.BGR, tof.FrameType.RADIAL, tof.FrameType.INTENSITY]
            # types = [tof.FrameType.BGR]
            tof.selectStreams(self.camera, types)  # type: ignore
            self.camera.start()
        except Exception as exc:
            log.exception("Could not open camera")
            raise exc
        log.info(f"Opened {self.name}")

    def get_single_image(self):
        self.get_cache_all_rgb_intensity_radial()
        image = resize_with_aspect_ratio(
            self.cached_image_rgb, width=DISPLAY_SIZE_WIDTH
        )
        return image

    def get_cache_all_rgb_intensity_radial(self):
        """
        Gets all (RGB, Intensity, Radial) frames from the camera and caches them.
        """
        if self.camera.isStreaming():
            start = timer()
            # Dispose of 10 images first. It appears that when we are not actively pulling frames often, the first few frames return rubbish.
            # This is done in many Chronoptics examples and anecdotally tested does help a lot.
            for _ in range(10):
                frames = self.camera.getFrames()
            frames = self.camera.getFrames()
            self.cached_image_time = time.strftime("%Y-%m-%dT%H%M%S")
            # See note about BGR / RGB
            self.cached_image_rgb = cv2.flip(
                np.asarray(getFrame(frames, tof.FrameType.BGR)), flipCode=-1
            )  # flip & mirror
            self.cached_image_intensity = np.asarray(
                getFrame(frames, tof.FrameType.INTENSITY)
            )
            self.cached_image_radial = np.asarray(
                getFrame(frames, tof.FrameType.RADIAL)
            )
            self.cached_image = self.cached_image_rgb  # For compatability
            end = timer()
            log.debug(f"Acquiring image(s) for {self.name} took: {end - start}")

    def get_normalised_intensity(self) -> cv2.Mat:
        if not hasattr(self, "cached_image_intensity"):
            raise ValueError("Must cache self.cached_image_intensity first")
        self.cached_image_intensity_normalised = np.copy(self.cached_image_intensity)
        # we will take "most" of the data and scale it to the ful range and "clip" the rest.
        gain_scalar = (
            255 / np.quantile(self.cached_image_intensity_normalised, 0.975)
        ).astype(np.uint8)
        if gain_scalar > 25:
            log.warning(f"get_normalised_intensity using high gain of {gain_scalar}")
        self.cached_image_intensity_normalised *= gain_scalar
        self.cached_image_intensity_normalised = np.clip(
            self.cached_image_intensity_normalised, 0, 255
        )
        return self.cached_image_intensity_normalised

