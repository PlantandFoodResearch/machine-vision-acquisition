from __future__ import annotations
import typing
import logging
import time
import gi

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machine_vision_acquisition_python.viewer.cli import CameraHelper


log = logging.getLogger(__name__)


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
    log.info(cam_stats)
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
        device.set_boolean_feature_value("PtpEnable", True)
