import cv2
import numpy as np
from machine_vision_acquisition_python.calibration.shared import (
    Calibration,
    CameraModel,
)
from machine_vision_acquisition_python.process.stereo.shared import (
    StereoParams,
    StereoProcessor,
    _marshal_point_to_array,
)
from numpy.typing import NDArray
from typing import Union, List, Tuple


class SparseStereoProcessor(StereoProcessor):
    """
    This processor allows you to work with sparse stereo depth calculations
    rather than entire images.

    Primarily it allows you to give pixel(u,v) coordinates of corresponding features
    in a left and right synchronised view and return stereo disparity or depth.
    """

    def __init__(
        self, calibration_left: Calibration, calibration_right: Calibration
    ) -> None:
        super().__init__(calibration_left, calibration_right)

    def undistort_image_points_l(self, image_points: NDArray) -> NDArray:
        """Return undistorted points for left camera"""
        if self.camera_model == CameraModel.OpenCV:
            function_undistort_points = cv2.undistortPoints
        elif self.camera_model == CameraModel.OpenCVFisheye:
            function_undistort_points = cv2.fisheye.undistortPoints
        else:
            raise NotImplementedError(
                f"Camera model {self.camera_model} not supported (yet!)"
            )

        undistorted_points = function_undistort_points(
            image_points,
            self.calibration_left.cameraMatrix,
            self.calibration_left.distCoeffs,
            R=self.params.R1,
            P=self.params.P1,
        )
        return undistorted_points

    def undistort_image_points_r(self, image_points: NDArray) -> NDArray:
        """Return undistorted points for right camera"""

        if self.camera_model == CameraModel.OpenCV:
            function_undistort_points = cv2.undistortPoints
        elif self.camera_model == CameraModel.OpenCVFisheye:
            function_undistort_points = cv2.fisheye.undistortPoints
        else:
            raise NotImplementedError(
                f"Camera model {self.camera_model} not supported (yet!)"
            )

        undistorted_points = function_undistort_points(
            image_points,
            self.calibration_right.cameraMatrix,
            self.calibration_right.distCoeffs,
            R=self.params.R2,
            P=self.params.P2,
        )
        return undistorted_points

    def disparity_from_dual_points(
        self,
        left_point: Union[NDArray, List],
        right_point: Union[NDArray, List],
        vertical_tolerance_px=10,
    ) -> float:
        """Given two points, return the horizontal disparity in pixel units"""
        left_points = (
            _marshal_point_to_array(left_point).reshape(-1, 1, 2).astype(np.float32)
        )  # Make a 1xN/Nx1 2-channel CV_32FC2 array
        right_points = (
            _marshal_point_to_array(right_point).reshape(-1, 1, 2).astype(np.float32)
        )

        # Undistort points
        left_point_undistorted = self.undistort_image_points_l(left_points)[0][0]
        right_point_undistorted = self.undistort_image_points_r(right_points)[0][0]

        # Disparity in dual dimensions
        disp = left_point_undistorted - right_point_undistorted

        # Check that Y pixels are within tolerances
        if abs(disp[1]) > vertical_tolerance_px:
            raise ValueError(
                f"Pixels are not vertically aligned within tolerance: {disp[1]}"
            )
        return disp[0]
