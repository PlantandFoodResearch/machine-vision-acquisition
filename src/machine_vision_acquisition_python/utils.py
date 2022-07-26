from __future__ import annotations
import typing
import logging
import time
import cv2
from pathlib import Path
import numpy as np
log = logging.getLogger(__name__)
try:
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
except ImportError as _:
    log.warning(
        f"Could not import Aravis, calling some functions will cause exceptions."
    )

try:
    import fpnge
    _USE_FPNGE = True
except ImportError as _:
    log.warning("python-fpnge not found: Using slow cv2.imwrite for saving PNG files")
    _USE_FPNGE = False

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machine_vision_acquisition_python.viewer.cli import CameraHelper


def save_png(path: Path, image: cv2.Mat, mkdir: bool = False) -> None:
    """Helper to attempt to use much faster fpnge, otherwise fallback to cv2"""
    if path.suffix.lower() != ".png":
        raise ValueError(f"File path must be a *.png")
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)  # Optionally always mkdir
    if _USE_FPNGE:
        image_bytes = fpnge.fromMat(image)
        if path.write_bytes(image_bytes) != len(image_bytes):
            raise IOError(f"Failed to write to {path}")
        return
    else:
        if not cv2.imwrite(str(path), image):
            raise IOError(f"Failed to write to {path}")
        return


def check_ptp_sync(cameras: typing.List[CameraHelper]):
    """Sets up PTP triggering and checks it is functional"""
    cam_stats = {}
    for camera in cameras:
        device: Aravis.Device = camera.camera.get_device()
        PtpStatus = device.get_feature("PtpStatus")
        PtpEnable = device.get_feature("PtpEnable")
        PtpClockAccuracy = device.get_feature("PtpClockAccuracy")
        cam_stats.update(
            {
                camera.name: {
                    "PtpEnable": PtpEnable.get_value_as_string(),
                    "PtpStatus": PtpStatus.get_value_as_string(),
                    "PtpClockAccuracy": PtpClockAccuracy.get_value_as_string(),
                }
            }
        )
    log.debug(cam_stats)
    return cam_stats


def enable_ptp_sync(cameras: typing.List[CameraHelper]):
    """Enable PtpEnable for all cameras in list and wait for state to settle"""
    for camera in cameras:
        device: Aravis.Device = camera.camera.get_device()
        device.set_boolean_feature_value("PtpEnable", True)
    check_ptp_sync(cameras)
    time.sleep(1.0)
    check_ptp_sync(cameras)
    pass


def toggle_device_ptp_sync(device: Aravis.Device):
    ptp_enable: bool = device.get_boolean_feature_value("PtpEnable")
    device.set_boolean_feature_value("PtpEnable", not ptp_enable)


def disable_ptp_sync(cameras: typing.List[CameraHelper]):
    """Disable PtpEnable for all cameras in a list"""
    for camera in cameras:
        device: Aravis.Device = camera.camera.get_device()
        device.set_boolean_feature_value("PtpEnable", False)


def get_image_sharpness(image: cv2.Mat, size=60):
    # Inspired by https://github.com/PlantandFoodResearch/Morphometrics/blob/master/morphometrics/frames/quality.py
    # Which was inspired by https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/

    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    h, w = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean


def get_image_max(image: cv2.Mat):
    max = np.iinfo(image.dtype.type).max  # type: ignore
    return float(np.max(image) / max)


def get_image_mean(image: cv2.Mat):
    """Gets mean image value, converting to gray if required first"""
    channels = image.shape[-1] if image.ndim == 3 else 1
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(image)
    max = np.iinfo(image.dtype.type).max  # type: ignore
    return float(mean / max)


def get_image_std(image: cv2.Mat):
    """Gets standard deviation image values, converting to gray if required first"""
    channels = image.shape[-1] if image.ndim == 3 else 1
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(image)
    max = np.iinfo(image.dtype.type).max  # type: ignore
    return float(std / max)
