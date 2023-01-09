from numpy.typing import NDArray
from enum import Enum


class CameraModel(Enum):
    OpenCV = "CameraModelOpenCV"
    OpenCVFisheye = "CameraModelOpenCVFisheye"


class Calibration:
    """
    A consistent representation of a Camera calibration
    """

    def __init__(
        self,
        name_or_serial,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        image_width,
        image_height,
        camera_model: CameraModel = CameraModel.OpenCV,
    ) -> None:
        self.serial: str = name_or_serial
        self.cameraMatrix: NDArray = cameraMatrix
        self.distCoeffs: NDArray = distCoeffs
        self.rvec: NDArray = rvec
        self.tvec: NDArray = tvec
        self.image_width: int = image_width
        self.image_height: int = image_height
        self.camera_model: CameraModel = camera_model

    @property
    def focallength_px(self):
        """Returns the camera's focal length in pixel units"""
        return float(self.cameraMatrix[0, 0])
