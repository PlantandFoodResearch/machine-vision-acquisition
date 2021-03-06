import click
import logging
import cv2
import os
import typing
import ctypes
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
import machine_vision_acquisition_python.utils
from machine_vision_acquisition_python.process.processing import (
    cvt_tonemap_image,
    resize_with_aspect_ratio,
)
from machine_vision_acquisition_python.interfaces.aravis import CameraHelper

log = logging.getLogger(__name__)
import gi

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis

try:
    from machine_vision_acquisition_python.interfaces.chronoptics import ToFCameraHelper
except ImportError as _:
    log.warning(f"Failed to import chronoptics")
    ToFCameraHelper = None
log = logging.getLogger(__name__)
try:
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
except ImportError as _:
    log.error(f"Could not import Aravis, calling most functions will cause exceptions.")

DISPLAY_SIZE_WIDTH = 1280


@click.command()
@click.option(
    "--name",
    "-n",
    help="Camera name to connect to. Defaults to None which will randomly select a camera",
    default=None,
)
@click.option(
    "--tof", "-t", help="Attempt to use Chronoptics camera", default=True, type=bool
)
@click.option(
    "--all",
    help="Open all cameras at once",
    default=False,
    is_flag=True,
)
@click.option(
    "--out-dir",
    "-o",
    help="Directory to save output too",
    default=Path("./tmp/"),
    type=click.Path(file_okay=False, resolve_path=True),
)
@click.option(
    "--factory-reset",
    help="Performs a factory reset only and then exits",
    default=False,
    is_flag=True,
)
def cli(name: str, all: bool, out_dir, factory_reset: bool, tof: bool):
    """
    Simple camera snapshot grabber and saver.
    Hotkeys:\n
    * 'q' to exit\n
    * 'e' to auto expose (run cameras for 5s each)\n
    * 'n' to grab new frames from cameras (software triggered, not synchronised)\n
    * 's' to save the displayed frame (as non-processed BayerRG12 PNG)\n
    * 't' to save the displayed frame (as tonemapped RGB PNG)\n
    * 'p' to toggle PTP mode (and print stats)\n
    """
    out_dir = Path(out_dir).resolve()  # Resolve and cast to Path
    out_dir.mkdir(exist_ok=True)

    # Check display aspects
    if os.name == "posix" and "DISPLAY" in os.environ and os.environ["DISPLAY"] != "":
        DISPLAY = True
    else:
        # Don't support windows at this stage
        DISPLAY = False

    cameras: typing.List[CameraHelper] = []
    if all:
        # Open all cameras
        Aravis.update_device_list()
        # Open all cameras
        for i in range(Aravis.get_n_devices()):
            try:
                camera = CameraHelper(Aravis.get_device_id(i))
            except Exception as exc:
                # Ignore non-compliant cameras (Chronoptics)
                if "arv-device-error-quark" in str(exc):
                    if tof:
                        serial = Aravis.get_device_serial_nbr(0)
                        camera = ToFCameraHelper(serial)
                    else:
                        continue
                else:
                    raise exc
            cameras.append(camera)
        # cameras = [CameraHelper(name=None) for i in range(Aravis.get_n_devices())]

    if not cameras:
        log.warning("Could not open any cameras, exiting!")
        return
    if factory_reset:
        for camera in cameras:
            try:
                # Try userset loading first
                camera.load_default_settings()
                log.info(f"Loaded default settings for {camera.name}")
                # camera.camera.execute_command("DeviceFactoryReset")  # type: ignore
                # log.info(f"DeviceFactoryReset {camera.name}")
            except Exception as exc:
                log.exception(f"Failed to reset {camera.name}", exc_info=exc)
                pass  # best effort
        return
    # Create the windows
    if DISPLAY:
        for camera in cameras:
            cv2.namedWindow(f"{camera.name}", cv2.WINDOW_NORMAL)

    try:
        snap_counter = 0
        while True:
            if DISPLAY:
                ch = cv2.waitKey(10) & 0xFF
                if ch == 27:
                    ch = "q"
                else:
                    ch = chr(ch)  # Convert to char
            else:
                ch = click.prompt(
                    "Please enter a command",
                    type=click.types.STRING,
                    show_choices=click.Choice(["n", "s", "t", "q"]),
                )
            if ch == "q":
                break
            elif ch == "e":
                for camera in cameras:
                    camera.settle_auto_exposure()
            elif ch == "p":
                machine_vision_acquisition_python.utils.enable_ptp_sync(cameras)
            elif ch == "n":
                # Capture new image from all cameras
                for camera in cameras:
                    image = camera.get_single_image()
                    if not DISPLAY:
                        continue
                    image = resize_with_aspect_ratio(image, width=DISPLAY_SIZE_WIDTH)
                    cv2.imshow(f"{camera.name}", image)
            elif ch == "s" or ch == "t":
                # save currently displaying image image
                # BUG: OpenCV cannot save 12b images (https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)
                # TODO: We must do the conversion ourselves :)
                snap_counter += 1
                for camera in cameras:
                    camera_dir = out_dir / camera.name
                    camera_dir.mkdir(exist_ok=True)
                    if camera.cached_image is None or camera.cached_image_time is None:
                        log.warning("Cannot save image yet, none cached!")
                        break
                    file_path = (
                        camera_dir
                        / f"{camera.cached_image_time}-{camera.name}-snapshot-{snap_counter}.png"
                    )
                    image = camera.cached_image
                    if tof and isinstance(camera, ToFCameraHelper):
                        # We handle this camera specially, since it is two cameras...
                        image_intensity = camera.get_normalised_intensity()
                        if image_intensity is not None:
                            # Save intensity image as a seperate camera
                            camera_dir_intensity = out_dir / f"{camera.name}-intensity"
                            file_path_intensity = (
                                camera_dir_intensity
                                / f"{camera.cached_image_time}-{camera.name}-snapshot-{snap_counter}.png"
                            )
                            camera_dir_intensity.mkdir(exist_ok=True)
                            if cv2.imwrite(str(file_path_intensity), image_intensity):
                                log.debug(f"Saved {file_path_intensity}")
                    elif ch == "t":
                        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
                        image = cvt_tonemap_image(image)
                        file_path = (
                            camera_dir
                            / f"{camera.cached_image_time}-{camera.name}-snapshot-tonemapped-{snap_counter}.png"
                        )
                    file_path.parent.mkdir(exist_ok=True)
                    if cv2.imwrite(str(file_path), image):
                        log.debug(f"Saved {file_path}")
    except SystemExit as _:
        pass  # CTRL-C
    finally:
        for camera in cameras:
            try:
                if tof:
                    if isinstance(camera, ToFCameraHelper):
                        camera.camera.stop()
                else:
                    camera.camera.stop_acquisition()
            except Exception as _:
                log.debug(f"Failed to stop {camera.name}")
        Aravis.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()
