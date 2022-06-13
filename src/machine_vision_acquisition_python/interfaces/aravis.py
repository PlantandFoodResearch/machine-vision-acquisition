import logging
import cv2
import time
import typing
import weakref
import threading
from timeit import default_timer as timer
import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis, GObject
import cv2
from machine_vision_acquisition_python.converter.processing import (
    cvt_tonemap_image,
    resize_with_aspect_ratio,
    unpack_BayerRG12,
    unpack_BayerRG12Packed,
)


log = logging.getLogger(__name__)

PIXEL_FORMAT_PREFERENCE_LIST = [
    "BayerRG12Packed",
    "BayerRG12",
    "BayerRG16",
    "BayerRG8",
]

class ArvStream(Aravis.Stream, GObject.GObject): pass


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
        log.debug(f"Opened {self.name}")
        # Attempt to ensure camera is stopped on exit
        self._finalizer = weakref.finalize(self, Aravis.Camera.stop_acquisition, self.camera)
        self.cached_image: typing.Optional[typing.Any] = None
        self.cached_image_time: typing.Optional[str] = None
        self.pixel_format_str: typing.Optional[str] = None
        self.lock = threading.Lock()
        self._frame_counter = 0
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
                log.debug(f"Using {self.pixel_format_str} for {self.name}")
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
        self.camera.set_frame_rate(5.0)  # type: ignore
        self.camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        # self.camera.set_trigger()
        # self.camera.set_exposure_time(1000)

        # self.camera.set_trigger("Software")  # type: ignore
        self.stream: ArvStream = self.camera.create_stream()  # type: ignore
        self.stream.connect("new-buffer", self._stream_cb, weakref.ref(self))
        payload = self.camera.get_payload()
        # allocate buffers
        for i in range(10):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.camera.start_acquisition()
        time.sleep(0.1)
        self.stream.set_emit_signals(True)  # type: ignore
        # Ensure they are stopped on destruction
        self._finalizer = weakref.finalize(self, Aravis.Stream.set_emit_signals, self.stream, False)  # type: ignore
        log.debug(f"main thread: {threading.get_ident()}")
        # self.camera.software_trigger()
        time.sleep(0.5)
        with self.lock:
            self._frame_counter = 0
            start = timer()
        # self.camera.software_trigger()
        time.sleep(5.0)
        with self.lock:
            end = timer()
            counter = self._frame_counter
            self._frame_counter = 0
        log.info(f"FPS: {counter / (end - start)}")
        time.sleep(1.0)



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

    @staticmethod
    def _stream_cb(stream: ArvStream, user_data: weakref.ReferenceType["CameraHelper"]):
        """This is called in the Aravis Thread. The debugger seems to not work here, so keep it simple!"""
        camera = user_data()
        if not camera:
            log.info(f"cb_called badref to camera, returning")
            return
        buffer: typing.Optional[Aravis.Buffer] = stream.try_pop_buffer()
        if not buffer:
            log.warning("_stream_cb failed to pop buffer")
            return
        try:
            if not buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                log.warning("_stream_cb failed to pop SUCCESS buffer")
                return
            # We have a proper buffer
            # log.info(f"Valid buffer received ({camera.name}) ts: {buffer.get_timestamp()}")
            with camera.lock:
                camera._frame_counter += 1
        finally:
            stream.push_buffer(buffer)  # type: ignore
        return


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
            if image is None or not image.any():
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


def get_camera_by_serial(serial: str) -> CameraHelper:
    """Relies on that on module load Aravis.update_device_list() is called"""
    for i in range(Aravis.get_n_devices()):
        if serial == str(Aravis.get_device_serial_nbr(i)):
            log.debug(f"Found device at address {Aravis.get_device_address(i)} to match serial {serial}")
            return CameraHelper(Aravis.get_device_id(i))
    raise ValueError(f"Could not find Aravis camera by serial: {serial}")


# Perform once on module load
Aravis.update_device_list()
