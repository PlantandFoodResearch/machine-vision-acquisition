from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
import cv2


class Calibration:
    def __init__(self, name_or_serial, cameraMatrix, distCoeffs, rvec, tvec) -> None:
        self.serial = name_or_serial
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.rvec = rvec
        self.tvec = tvec


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

        self._optimal_matrix = cv2.UMat(np.ndarray((3, 3), np.float64, newcameramtx))
        self._roi = roi

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
    def initialised(self):
        if self._optimal_matrix is None or self._roi is None:
            return False
        return True

    def undistort(self, image: cv2.Mat, crop = True) -> cv2.Mat:
        result = cv2.undistort(
            image,
            self.calibration.cameraMatrix,
            self.calibration.distCoeffs,
        )
        if crop:
            y, x, h, w = self.roi
            result = result[y : y + h, x : x + w]
        return result
