"""
Provides a python interface to libcalib: https://downloads.calib.io/libcalib/doc/
"""
import json
import logging
from pathlib import Path
from typing import List
from machine_vision_acquisition_python.calibration.shared import Calibration
import numpy as np


log = logging.getLogger(__name__)


def load_from_calibio_json(calibio_json: Path) -> List[Calibration]:
    """
    Reads the calibio JSON file output and returns all camera calibrations contained.

    Based off: https://downloads.calib.io/libcalib/doc/a01086.html
    """
    if not calibio_json.exists():
        raise ValueError(f"File doesn't exist: {calibio_json}")
    calib = json.loads(calibio_json.read_text())
    valid_calibs = []
    for camera in calib["Calibration"]["cameras"]:
        try:
            polymorphic_name = camera["model"].get("polymorphic_name")
            polymorphic_id = camera["model"].get("polymorphic_id")
            # for some reason CalibIO outptus polymorphic_id==1 for 'other' cameras (not the first)
            if polymorphic_name is None and polymorphic_id != 1:
                log.warning(
                    f"Skipping invalid camera type, polymorphic_id: {camera['model']['polymorphic_id']}"
                )
                continue
            else:
                if polymorphic_name != "libCalib::CameraModelOpenCV":
                    log.debug(f"polymorphic_name: {polymorphic_name}")
                if polymorphic_id == 1:
                    log.debug(
                        f"Using polymorphic_id==1 for camera serial {camera.get('serial', 'unknown')}"
                    )
                valid_calibs.append(camera)
        except:
            log.exception(f"Skipping invalid camera, unknown error")
            continue

    camera_calibrations: List[Calibration] = []
    camera_count = len(valid_calibs)
    log.debug(f"reading {camera_count} camera calibrations from {calibio_json}")

    for camera in valid_calibs:
        intrinsics = camera["model"]["ptr_wrapper"]["data"]["parameters"]
        cam_matrix, distortion_matrix = read_camera_intrinsics(intrinsics)
        transform = camera["transform"]
        rvec, tvec = read_camera_extrinsics(transform)
        image_size = camera["model"]["ptr_wrapper"]["data"]["CameraModelCRT"]["CameraModelBase"]["imageSize"]
        #TODO add imageSize here
        # todo: get serial mappings
        serial = camera.get("serial", "unknown")
        if serial == "unknown":
            log.warning(f"calibration does not reference camera serial, is it present?")
        calib = Calibration(serial, cam_matrix, distortion_matrix, rvec, tvec, image_size["width"], image_size["height"])
        # Use openCV naming
        camera_calibrations.append(calib)
    return camera_calibrations


def read_camera_intrinsics(intrinsics: dict):
    """See https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a for matrix details"""
    f = intrinsics["f"]["val"]
    ar = intrinsics["ar"]["val"]
    cx = intrinsics["cx"]["val"]
    cy = intrinsics["cy"]["val"]
    k1 = intrinsics["k1"]["val"]
    k2 = intrinsics["k2"]["val"]
    k3 = intrinsics["k3"]["val"]
    k4 = intrinsics["k4"]["val"]
    k5 = intrinsics["k5"]["val"]
    k6 = intrinsics["k6"]["val"]
    p1 = intrinsics["p1"]["val"]
    p2 = intrinsics["p2"]["val"]
    s1 = intrinsics["s1"]["val"]
    s2 = intrinsics["s2"]["val"]
    s3 = intrinsics["s3"]["val"]
    s4 = intrinsics["s4"]["val"]
    tauX = intrinsics["tauX"]["val"]
    tauY = intrinsics["tauY"]["val"]
    tmp = [[f, 0.0, cx], [0.0, f * ar, cy], [0.0, 0.0, 1.0]]
    cam_matrix = np.matrix(tmp, dtype=np.float64)
    distortion_matrix = np.matrix(
        [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY], dtype=np.float64
    )
    return cam_matrix, distortion_matrix


def read_camera_extrinsics(transform: dict):
    """
    See https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a for matrix details
    Note: Our ourput did not contain qw, qx, qy, qz, but instead rx, ry, rz.
    """
    q = transform["rotation"]

    rx = q["rx"]
    ry = q["ry"]
    rz = q["rz"]

    rvec = np.matrix([rx, ry, rz])

    t = transform["translation"]
    tvec = [t["x"], t["y"], t["z"]]
    tvec = np.matrix(tvec)
    # Convert tvec from meters to mm
    tvec = tvec * 1000
    return rvec, tvec
