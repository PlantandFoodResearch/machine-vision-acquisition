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

    def mm_per_px_at_z(self, depth_mm: float) -> float:
        """
        Return mm per pixel at a given depth (in mm) using this camera's calibration.

        Based on x_px=(f/Z)*x_mm,
        with x_px == 1 and Z == depth_plane
        """
        f = self.cameraMatrix[0, 0]
        return depth_mm / f
