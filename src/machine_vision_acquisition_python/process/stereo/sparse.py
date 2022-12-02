import cv2
import numpy as np
from machine_vision_acquisition_python.calibration.shared import Calibration
from machine_vision_acquisition_python.process.stereo.shared import StereoParams, StereoProcessor
from numpy.typing import NDArray
from typing import Union, List


def _marshal_point_to_array(point: Union[NDArray, List]):
    if isinstance(point, list):
        point = np.array(point).astype(np.float32)
    return point


class SparseStereoProcessor(StereoProcessor):
    """
    This processor allows you to work with sparse stereo depth calculations
    rather than entire images.

    Primarily it allows you to give pixel(u,v) coordinates of corresponding features
    in a left and right synchronised view and return stereo disparity or depth.
    """
    def __init__(self, calibration_left: Calibration, calibration_right: Calibration) -> None:
        super().__init__(calibration_left, calibration_right)


    def undistort_image_points_l(self, image_points: NDArray) -> NDArray:
        """Return undistorted points for left camera"""
        undistorted_points = cv2.undistortPoints(
            image_points,
            self.calibration_left.cameraMatrix,
            self.calibration_left.distCoeffs,
            R=self.params.R1,
            P=self.params.P1)
        return undistorted_points


    def undistort_image_points_r(self, image_points: NDArray) -> NDArray:
        """Return undistorted points for right camera"""
        undistorted_points = cv2.undistortPoints(
            image_points,
            self.calibration_right.cameraMatrix,
            self.calibration_right.distCoeffs,
            R=self.params.R2,
            P=self.params.P2)
        return undistorted_points


    def disparity_from_dual_points(self, left_point: Union[NDArray, List], right_point: Union[NDArray, List], vertical_tolerance_px=10) -> float:
        left_points = _marshal_point_to_array(left_point).reshape(1,2).astype(np.float32)
        right_points = _marshal_point_to_array(right_point).reshape(1,2).astype(np.float32)
        
        # Undistort points
        left_point_undistorted = self.undistort_image_points_l(left_points)[0][0]
        right_point_undistorted = self.undistort_image_points_r(right_points)[0][0]

        # Disparity in dual dimensions
        disp = left_point_undistorted - right_point_undistorted

        # Check that Y pixels are within tolerances
        if abs(disp[1]) > vertical_tolerance_px:
            raise ValueError(f"Pixels are not vertically aligned within tolerance: {disp[1]}") 
        return disp[0]