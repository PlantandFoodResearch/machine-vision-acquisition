"""
Python interface to the Aravis GenICam library.
"""
import logging
import cv2
import time
import typing
import weakref
import threading
import pandas as pd
import numpy as np
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
from machine_vision_acquisition_python.models import GenICamParam


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

PIXEL_FORMAT_PREFERENCE_LIST = [
    "BayerRG12Packed",
    "BayerRG12",
    "BayerRG16",
    "BayerRG8",
]
_MAX_PARAM_WRITE_ATTEMPTS = 2

class ArvStream(Aravis.Stream, GObject.GObject): pass


def convert_with_lock(buf: Aravis.Buffer) -> typing.Optional[cv2.Mat]:
    """Convert an Aravis.Buffer to cv2.Mat, whilst the caller MUST hold a lock on the buffer"""
    ### Credit: https://github.com/SintefRaufossManufacturing/python-aravis/blob/master/aravis.py#L181
    if not buf:
        return None
    raw_data_ptr = buf.get_data()  # type: ignore
    pixel_format = buf.get_image_pixel_format()
    height = buf.get_image_height()
    width = buf.get_image_width()
    if pixel_format == Aravis.PIXEL_FORMAT_BAYER_RG_12_PACKED:
        image = unpack_BayerRG12Packed(raw_data_ptr, height, width)
    elif pixel_format == Aravis.PIXEL_FORMAT_BAYER_RG_12:
        image = unpack_BayerRG12(raw_data_ptr, height, width)
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")
    # return a copy so that buffer memory can be released without causing havoc
    return image.copy()


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
        self.cached_image_time: typing.Optional[np.datetime64] = None
        self.pixel_format_str: typing.Optional[str] = None
        self.lock = threading.RLock()
        self._frame_counter = 0
        self.latest_buffer: typing.Optional[Aravis.Buffer] = None
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
        self.select_pixel_format()

        # self.camera.set_frame_rate(5.0)  # type: ignore
        # self.camera.set_acquisition_mode(Aravis.AcquisitionMode.SINGLE_FRAME)
        # self.camera.set_trigger("Software")  # type: ignore
        # self.camera.set_exposure_time(1000)

        if self.camera.is_gv_device():
            self.camera.gv_set_packet_size_adjustment(
                Aravis.GvPacketSizeAdjustment.ON_FAILURE
            )
            self.camera.gv_set_packet_size(self.camera.gv_auto_packet_size())


        self.stream: ArvStream = self.camera.create_stream()  # type: ignore
        self.stream.connect("new-buffer", self._stream_buffer_new_cb, weakref.ref(self))
        self.stream.set_emit_signals(True)  # type: ignore
        payload = self.camera.get_payload()
        # allocate buffers
        for i in range(10):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        # Ensure they are stopped on destruction
        self._finalizer = weakref.finalize(self, Aravis.Stream.set_emit_signals, self.stream, False)  # type: ignore
        log.debug(f"setup stream and cb_processing for {self.name}")


    def calc_fps_blocking(self, blocking_time_s = 1.0):
        with self.lock:
            self._frame_counter = 0
            start = timer()
        time.sleep(blocking_time_s)
        with self.lock:
            end = timer()
            counter = self._frame_counter
            self._frame_counter = 0
        log.debug(f"FPS: {counter / (end - start)}")


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
    def _stream_buffer_new_cb(stream: ArvStream, user_data: weakref.ReferenceType["CameraHelper"]):
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
            log.debug(f"Valid buffer received ({camera.name}) ts(device): {buffer.get_timestamp()}, ts(system): {buffer.get_system_timestamp()}")
            # Interact with camera object in thread-safe manner. Swaps out latest buffer and increments buffer count
            previous_buffer = None
            with camera.lock:
                camera._frame_counter += 1
                previous_buffer = camera.latest_buffer
                camera.latest_buffer = buffer
            if previous_buffer is not None:
                stream.push_buffer(camera.latest_buffer)  # type: ignore
        except Exception as _:
            stream.push_buffer(buffer)  # type: ignore
            raise
        return

    def start_capturing(self):
        """Best effort of getting images started...."""
        self.camera.start_acquisition()
        try:
            self.camera.software_trigger()
        except Exception as _:
            pass

    def set_parameter(self, param: GenICamParam) -> None:
        """Attempts to set GenICam parameters"""
        arv_device: Aravis.Device = self.camera.get_device()
        param_type = param.val_type or type(param.value)
        param_type_cast = param_type(param.value)  # type: ignore

        if not arv_device.is_feature_available(param.name):  # type: ignore
            raise ValueError(f"Feature {param.name} not availible on {self.name}")
        feature_access_mode: Aravis.GcAccessMode = arv_device.get_feature(param.name).get_actual_access_mode()  # type: ignore
        if feature_access_mode is not Aravis.GcAccessMode.RW and feature_access_mode is not Aravis.GcAccessMode.WO:
            raise AttributeError(f"Feature {param.name} access mode is {feature_access_mode}")

        attempts = 0
        while True:
            try:
                if param_type == str:
                    arv_device.set_string_feature_value(param.name, str(param.value))  # type: ignore
                elif param_type == bool:
                    arv_device.set_boolean_feature_value(param.name, bool(param.value))  # type: ignore
                elif param_type == float:
                    arv_device.set_float_feature_value(param.name, float(param.value))  # type: ignore
                elif param_type == int:
                    arv_device.set_integer_feature_value(param.name, int(param.value))  # type: ignore
                else:
                    try:
                        arv_device.set_features_from_string(f"{param.name}={param.value}")  # type: ignore
                        break
                    except Exception as _:
                        raise ValueError(f"Could not set {param.name}, unsupported type: {param_type}")
                break
            except Exception as exc:
                attempts += 1
                if "GigEVision write_memory timeout" in str(exc) and attempts < _MAX_PARAM_WRITE_ATTEMPTS:
                    # try again
                    log.warning(f"GigEVision write_memory timeout setting {param.name} on {self.name}, trying again...")
                    continue
                raise
        log.info(f"Set {param.name}={param.value} on {self.name}")

    def get_single_image(self):
        """Acquire and cache a single image"""
        raise NotImplementedError("Needs recreating")
        start = timer()

        self.camera.software_trigger()
        buffer = self.stream.timeout_pop_buffer(1 * 1000 * 1000)  # type: ignore # 1 second
        try:
            if not buffer:
                raise TimeoutError("Failed to get an image from the camera")
            image = convert_with_lock(buffer)
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

    def unpack_last_buffer(self):
        """Processes the latest Aravis Buffer into a OpenCV Mat"""
        if self.latest_buffer is None:
            raise ValueError("No buffer captured yet, has the camera been triggered?")
        with self.lock:
            self.cached_image = convert_with_lock(self.latest_buffer)
            self.cached_image_time = pd.Timestamp(self.latest_buffer.get_system_timestamp(), unit="ns").to_datetime64()


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
