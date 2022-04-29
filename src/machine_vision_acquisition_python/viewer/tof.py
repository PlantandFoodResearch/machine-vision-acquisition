import chronoptics.tof as tof
import cv2
import typing
import logging
import numpy as np
import time
import click
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
            tof.selectStreams(
                self.camera, types  # type: ignore
            )
            self.camera.start()
        except Exception as exc:
            log.exception("Could not open camera")
            raise exc
        log.info(f"Opened {self.name}")

    def get_single_image(self):
        self.get_cache_all_rgb_intensity_radial()
        image = resize_with_aspect_ratio(self.cached_image_rgb, width=DISPLAY_SIZE_WIDTH)
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
            self.cached_image_rgb = cv2.flip(np.asarray(getFrame(frames, tof.FrameType.BGR)), flipCode=-1)  # flip & mirror
            self.cached_image_intensity = np.asarray(getFrame(frames, tof.FrameType.INTENSITY))
            self.cached_image_radial = np.asarray(getFrame(frames, tof.FrameType.RADIAL))
            self.cached_image = self.cached_image_rgb  # For compatability
            end = timer()
            log.debug(f"Acquiring image(s) for {self.name} took: {end - start}")

    def get_normalised_intensity(self) -> cv2.Mat:
        if not hasattr(self, "cached_image_intensity"):
            raise ValueError("Must cache self.cached_image_intensity first")
        self.cached_image_intensity_normalised = np.copy(self.cached_image_intensity)
        # we will take "most" of the data and scale it to the ful range and "clip" the rest.
        gain_scalar = (255/np.quantile(self.cached_image_intensity_normalised, 0.975)).astype(np.uint8)
        if gain_scalar > 25:
            log.warning(f"get_normalised_intensity using high gain of {gain_scalar}")
        self.cached_image_intensity_normalised *= gain_scalar
        self.cached_image_intensity_normalised = np.clip(self.cached_image_intensity_normalised, 0, 255)
        return self.cached_image_intensity_normalised


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
    "--serial",
    help="KeaCamera serial to try",
    default=None,
    required=False,
    type=str
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