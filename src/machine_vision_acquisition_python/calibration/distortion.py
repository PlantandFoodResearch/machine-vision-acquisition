from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np
import numpy.typing as npt
from pathlib import Path
import cv2

import cv2
import numpy as np
from numpy.typing import NDArray
from machine_vision_acquisition_python.calibration.libcalib import Calibration


def write_opencv_yaml(self, filepath: Path):
    """Writes out these calibration values in the opencv YAML format"""
    fs = cv2.FileStorage(
        str(filepath), cv2.FILE_STORAGE_APPEND | cv2.FILE_STORAGE_FORMAT_YAML
    )
    try:
        fs.write(f"M-{self.serial}", self.cameraMatrix)
        fs.write(f"D-{self.serial}", self.distCoeffs)
        fs.write(f"R-{self.serial}", self.rvec)
        fs.write(f"T-{self.serial}", self.tvec)
    finally:
        fs.release()


def read_opencv_yaml(self, filepath: Path):
    raise NotImplementedError()


class Undistorter:
    def __init__(
        self, calibration: Calibration, image_shape: Optional[Tuple[int, int]] = None
    ) -> None:
        self.calibration = calibration
        self.image_shape = image_shape
        self._optimal_matrix = None
        self._roi: Optional[Tuple[int, int, int, int]] = None
        if self.image_shape is not None:
            self.init_optimal_matrix(self.image_shape)

    def init_optimal_matrix(self, shape: Tuple[int, int], alpha=1):
        """
        Note for alpha (from https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6):
            Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1.
            alpha=0 means that the rectified images are
            zoomed and shifted so that only valid pixels are visible (no black areas after rectification).

            alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the
            original images from the cameras are retained in the rectified images (no source image pixels are lost).

            Any intermediate value yields an intermediate result between those two extreme cases.
        """
        shape = (shape[0], shape[1])  # drop any potential channel information
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.calibration.cameraMatrix,
            self.calibration.distCoeffs,
            shape,
            alpha,
            shape,
        )
        if newcameramtx is None or roi is None:
            raise ValueError(f"Unable to compute optimal matrix")
        # fix up some type casting

        self._optimal_matrix = newcameramtx
        self._roi = roi
        y, x, h, w = self.roi
        # OpenCV is inconsistent with width, height or height, width order often :(
        self._undistorted_size = (w - x, h - y)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.calibration.cameraMatrix,
            self.calibration.distCoeffs,
            None,
            newcameramtx,
            self._undistorted_size,
            cv2.CV_32FC1,
        )
        pass

    @property
    def optimal_matrix(self):
        if self._optimal_matrix is None:
            raise ValueError(
                "init_optimal_matrix must be called before this can be used"
            )
        return self._optimal_matrix

    @property
    def roi(self):
        if self._roi is None:
            raise ValueError(
                "init_optimal_matrix must be called before this can be used"
            )
        return self._roi

    @property
    def map1(self):
        if self._map1 is None:
            raise ValueError(
                "init_optimal_matrix must be called before this can be used"
            )
        return self._map1

    @property
    def map2(self):
        if self._map2 is None:
            raise ValueError(
                "init_optimal_matrix must be called before this can be used"
            )
        return self._map2

    @property
    def initialised(self):
        if self._optimal_matrix is None or self._roi is None:
            return False
        return True

    def undistort(self, image: cv2.Mat, crop=True) -> cv2.Mat:
        result = cv2.remap(
            image,
            self.map1,
            self.map2,
            cv2.INTER_LINEAR,
        )
        if crop:
            x, y, h, w = self.roi
            result = result[y : y + h, x : x + w]
        return result
