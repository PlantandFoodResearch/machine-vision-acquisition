import click
import logging
import cv2
import time
import typing
from pathlib import Path
from timeit import default_timer as timer

log = logging.getLogger(__name__)


# temp
import gi

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import cv2
import ctypes
import numpy as np


def convert(buf):
    ### Credit: https://github.com/SintefRaufossManufacturing/python-aravis/blob/master/aravis.py#L181
    if not buf:
        return None
    pixel_format = buf.get_image_pixel_format()
    bits_per_pixel = pixel_format >> 16 & 0xFF
    if bits_per_pixel == 8:
        INTP = ctypes.POINTER(ctypes.c_uint8)
    else:
        INTP = ctypes.POINTER(ctypes.c_uint16)
    addr = buf.get_data()
    ptr = ctypes.cast(addr, INTP)
    im = np.ctypeslib.as_array(ptr, (buf.get_image_height(), buf.get_image_width()))
    im = im.copy()
    return im


def resize_with_aspect_ratio(
    image,
    width: "typing.Optional[int]" = None,
    height: "typing.Optional[int]" = None,
    inter=cv2.INTER_AREA,
):
    # borrowed from https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
    # And logc improved
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        # return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        return image
    if width is None and height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif width is not None and height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        raise ValueError("Cannot specify width and height")
    # cv2.resize(image, dim, interpolation=inter)
    # return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    return cv2.resize(image, dim, interpolation=inter)



class CameraHelper():
    def __init__(self, name: typing.Optional[str]) -> None:
        try:
            self.camera: Aravis.Camera = Aravis.Camera.new(name)
        except Exception as exc:
            log.exception("Could not open camera")
            raise exc
        self.name = f"{self.camera.get_model_name()}-{self.camera.get_device_serial_number()}"
        log.info(f"Opened {self.name}")
        self.cached_image: typing.Optional[typing.Any] = None
        self.cached_image_time: typing.Optional[str] = None
        self.set_default_camera_options()
    
    def set_default_camera_options(self):
        self.camera.gv_set_packet_size_adjustment(Aravis.GvPacketSizeAdjustment.ON_FAILURE)
        self.camera.gv_set_packet_size(self.camera.gv_auto_packet_size())
        self.camera.set_pixel_format(Aravis.PIXEL_FORMAT_BAYER_RG_12)
        self.camera.set_frame_rate(1)

        self.camera.set_trigger("Software")

        self.stream: Aravis.Stream = self.camera.create_stream(None, None)
        payload = self.camera.get_payload()
        self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    def get_single_image(self):
        """Acquire and cache a single image"""
        self.camera.start_acquisition()
        try:
            start = timer()
            self.camera.software_trigger()
            buffer = self.stream.timeout_pop_buffer(1 * 1000 * 1000)  # 1 second
            if not buffer:
                raise TimeoutError("Failed to get an image from the camera")
            image = convert(buffer)
            self.stream.push_buffer(buffer)  # push buffer back into stream
            self.cached_image = image  # Cache the raw image for optional saving
            self.cached_image_time = time.strftime("%Y-%m-%dT%H%M%S")
            image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
            image = resize_with_aspect_ratio(image, width=640)
            end = timer()
            log.debug(f"Acquiring image for {self.name} took: {end - start}")
        finally:
            self.camera.stop_acquisition()
        return image

    def get_last_image(self):
        """Returns previously acquired cached image"""
        try:
            return self.cached_image
        except AttributeError as _:
            return None


@click.command()
@click.option(
    "--name",
    "-n",
    help="Camera name to connect to. Defaults to None which will randomly select a camera",
    default=None,
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
def cli(name: str, all: bool, out_dir, factory_reset: bool):
    """
    Simple camera snapshot grabber and saver.
    Hotkeys:\n
    * 'q' to exit\n
    * 'n' to grab new frames from cameras (software triggered, not synchronised)\n
    * 's' to save the displayed frame (as non-processed BayerRG12 PNG)\n
    """
    out_dir = Path(out_dir).resolve()  # Resolve and cast to Path
    out_dir.mkdir(exist_ok=True)
    cameras: typing.List[CameraHelper] = []
    if all:
        # Open all cameras
        Aravis.update_device_list()
        # Open all cameras
        for i in range(Aravis.get_n_devices()):
            try:
                camera = CameraHelper(Aravis.get_device_address(i))
            except Exception as exc:
                # Ignore non-compliant cameras (Chronoptics)
                if "arv-device-error-quark" in str(exc):
                    continue
                raise exc
            cameras.append(camera)
        # cameras = [CameraHelper(name=None) for i in range(Aravis.get_n_devices())]
    if factory_reset:
        for camera in cameras:
            try:
                camera.camera.execute_command("DeviceFactoryReset")
                log.info(f"DeviceFactoryReset {camera.name}")
            except Exception as exc:
                log.exception(f"Failed to reset {camera.name}", exc_info=exc)
                pass  # best effort
        return
    # Create the windows
    for camera in cameras:
        cv2.namedWindow(f"{camera.name}", cv2.WINDOW_NORMAL)

    try:
        snap_counter = 0
        while True:
            ch = cv2.waitKey(10) & 0xFF
            if ch == 27 or ch == ord("q"):
                break
            elif ch == ord("n"):
                # Capture new image from all cameras
                for camera in cameras:
                    image = camera.get_single_image()
                    cv2.imshow(f"{camera.name}", image)
            elif ch == ord("s"):
                # save currently displaying image image
                snap_counter += 1
                for camera in cameras:
                    if camera.cached_image is None or camera.cached_image_time is None:
                        log.warning("Cannot save image yet, none cached!")
                        break
                    file_path = out_dir / f"{camera.cached_image_time}-{camera.name}-snapshot-{snap_counter}.png"
                    cv2.imwrite(str(file_path), camera.cached_image)
                    log.debug(f"Saved {file_path}")
    except SystemExit as _:
        pass  # CTRL-C
    finally:
        Aravis.shutdown()
        cv2.destroyAllWindows()

    # try:
    #     camera: Aravis.Camera = Aravis.Camera.new(name)
    # except Exception as exc:
    #     log.exception("Could not open camera")
    #     raise exc
    # log.info(f"Opened {camera.get_model_name()} {camera.get_device_serial_number()}")
    # camera.gv_set_packet_size_adjustment(Aravis.GvPacketSizeAdjustment.ON_FAILURE)
    # camera.gv_set_packet_size(camera.gv_auto_packet_size())
    # camera.set_frame_rate(1)
    # camera.set_trigger("Software")
    # stream = camera.create_stream(None, None)
    # payload = camera.get_payload()

    # # Ensure we have frames to buffer
    # for i in range(0, 1):
    #     stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    # log.info("starting acquisition")
    # camera.start_acquisition()
    # try:
    #     cv2.namedWindow("frame")
    #     while True:
    #         start = timer()
    #         ch = cv2.waitKey(10) & 0xFF
    #         if ch == 27 or ch == ord("q"):
    #             break
    #         elif ch == ord("n"):
    #             # cv2.imwrite("imagename.png", image)
    #             buffer = stream.try_pop_buffer()
    #             if buffer:
    #                 image = convert(buffer)
    #                 stream.push_buffer(buffer)  # push buffer back into stream
    #                 image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    #                 image = resize_with_aspect_ratio(image, width=640)
    #                 cv2.imshow("frame", image)
    #                 end = timer()
    #                 log.debug(f"Display time: {end - start}")
    # finally:
    #     camera.stop_acquisition()
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()
