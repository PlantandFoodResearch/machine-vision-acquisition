from numpy.typing import NDArray

class Calibration:
    """
    A consistent representation of a Camera calibration
    """
    def __init__(self, name_or_serial, cameraMatrix, distCoeffs, rvec, tvec, image_width, image_height) -> None:
        self.serial: str = name_or_serial
        self.cameraMatrix: NDArray = cameraMatrix
        self.distCoeffs: NDArray = distCoeffs
        self.rvec: NDArray = rvec
        self.tvec: NDArray = tvec
        self.image_width: int = image_width
        self.image_height: int = image_height