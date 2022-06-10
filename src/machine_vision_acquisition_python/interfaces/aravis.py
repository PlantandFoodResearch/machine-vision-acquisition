import logging
import cv2
import time
import typing
from timeit import default_timer as timer
import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import cv2
from machine_vision_acquisition_python.converter.processing import (
    cvt_tonemap_image,
    resize_with_aspect_ratio,
    unpack_BayerRG12,
    unpack_BayerRG12Packed,
)


log = logging.getLogger(__name__)

PIXEL_FORMAT_PREFERENCE_LIST = [
    "BayerRG12",
    "BayerRG12Packed",
    "BayerRG16",
    "BayerRG8",
]




def convert(buf) -> typing.Optional[cv2.Mat]:
    ### Credit: https://github.com/SintefRaufossManufacturing/python-aravis/blob/master/aravis.py#L181
    if not buf:
        return None
    pixel_format = buf.get_image_pixel_format()
    if pixel_format == Aravis.PIXEL_FORMAT_BAYER_RG_12_PACKED:
        image = unpack_BayerRG12Packed(buf)
    elif pixel_format == Aravis.PIXEL_FORMAT_BAYER_RG_12:
        image = unpack_BayerRG12(buf)
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")
    return image


class CameraHelper:
    def __init__(self, name: typing.Optional[str]) -> None:
        try:
            self.camera: Aravis.Camera = Aravis.Camera.new(name)  # type: ignore
        except Exception as exc:
            log.debug(f"Could not open camera: {name}")
            raise exc
        self.name = (
            f"{self.camera.get_model_name()}-{self.camera.get_device_serial_number()}"
        )
        log.info(f"Opened {self.name}")
        self.cached_image: typing.Optional[typing.Any] = None
        self.cached_image_time: typing.Optional[str] = None
        self.pixel_format_str: typing.Optional[str] = None
        self.set_default_camera_options()

    def select_pixel_format(self):
        """Attempt to use best option pixel format"""
        supported_pixel_formats_str: typing.List[
            str
        ] = self.camera.dup_available_pixel_formats_as_strings()
        for pixel_format_str in PIXEL_FORMAT_PREFERENCE_LIST:
            # TODO: Possibly do data sanitation
            if pixel_format_str in supported_pixel_formats_str:
                self.camera.set_pixel_format_from_string(pixel_format_str)  # type: ignore
                self.pixel_format_str = pixel_format_str
                log.info(f"Using {self.pixel_format_str} for {self.name}")
                return
        raise ValueError(
            f"Could not find a compatible pixel format in devices supported formats: {supported_pixel_formats_str}"
        )

    def set_default_camera_options(self):
        if self.camera.is_gv_device():
            self.camera.gv_set_packet_size_adjustment(
                Aravis.GvPacketSizeAdjustment.ON_FAILURE
            )
            self.camera.gv_set_packet_size(self.camera.gv_auto_packet_size())
        self.select_pixel_format()
        self.camera.set_frame_rate(30)
        # self.camera.set_exposure_time(1000)

        self.camera.set_trigger("Software")  # type: ignore

        self.stream: Aravis.Stream = self.camera.create_stream(None, None)  # type: ignore
        payload = self.camera.get_payload()
        self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.camera.start_acquisition()

    def settle_auto_exposure(self, length_s=5):
        """Run camera for x seconds to let exposure settle"""
        start = timer()
        try:
            while True:
                self.camera.software_trigger()
                buffer = self.stream.try_pop_buffer()
                if buffer is not None:
                    self.stream.push_buffer(buffer)
                if (timer() - start) > length_s:
                    break
            log.info(f"Ran camera for {length_s}s")
        except AttributeError as _:
            log.info(f"Could not run settle_auto_exposure for {self.name}")

    def get_single_image(self):
        """Acquire and cache a single image"""
        start = timer()
        #  grab and throw away 10 frames first for auto-exposure to settle.
        # for i in range(10):
        #     # dispose of some frames
        #     self.camera.software_trigger()
        #     buffer = self.stream.pop_buffer()
        #     if buffer is not None:
        #         self.stream.push_buffer(buffer)
        self.camera.software_trigger()
        buffer = self.stream.timeout_pop_buffer(1 * 1000 * 1000)  # type: ignore # 1 second
        try:
            if not buffer:
                raise TimeoutError("Failed to get an image from the camera")
            image = convert(buffer)
            if image is not None and not image.any():
                raise ValueError("Failed to convert buffer to image")
        finally:
            if buffer is not None:
                self.stream.push_buffer(buffer)  # push buffer back into stream
        self.cached_image = image  # Cache the raw image for optional saving
        self.cached_image_time = time.strftime("%Y-%m-%dT%H%M%S")

        image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        image = cvt_tonemap_image(image)

        end = timer()
        log.debug(f"Acquiring image for {self.name} took: {end - start}")
        return image

    def get_last_image(self):
        """Returns previously acquired cached image"""
        try:
            return self.cached_image
        except AttributeError as _:
            return None
