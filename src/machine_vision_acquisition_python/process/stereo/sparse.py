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

    def undistort_image_points(self, image_points: Union[NDArray, List], left: bool) -> NDArray:
        """Return undistorted points for left/right camera"""
        if self.camera_model == CameraModel.OpenCV:
            function_undistort_points = cv2.undistortPoints
        elif self.camera_model == CameraModel.OpenCVFisheye:
            function_undistort_points = cv2.fisheye.undistortPoints
        else:
            raise NotImplementedError(
                f"Camera model {self.camera_model} not supported (yet!)"
            )
        
        # Make a 1xN/Nx1 2-channel CV_32FC2 array from list
        if isinstance(image_points, list) or isinstance(image_points, tuple):
            image_points = (
                _marshal_point_to_array(image_points).reshape(-1, 1, 2).astype(np.float32)
            )
        else:
            # do nothing for now. Could maybe check NDArray shape to ensure compatible?
            pass

        if left:
            undistorted_points = function_undistort_points(
                image_points,
                self.calibration_left.cameraMatrix,
                self.calibration_left.distCoeffs,
                R=self.params.R1,
                P=self.params.P1,
            )
        else:
            undistorted_points = function_undistort_points(
                image_points,
                self.calibration_right.cameraMatrix,
                self.calibration_right.distCoeffs,
                R=self.params.R2,
                P=self.params.P2,
            )
        return np.squeeze(undistorted_points)

    def disparity_from_dual_points(
        self,
        left_point: Union[NDArray, List],
        right_point: Union[NDArray, List],
        vertical_tolerance_px=10,
    ) -> float:
        """Given two points, return the horizontal disparity in pixel units"""
        # Undistort points
        left_point_undistorted = self.undistort_image_points(left_point, left=True)
        right_point_undistorted = self.undistort_image_points(right_point, left=False)

        # Disparity in dual dimensions
        disp = left_point_undistorted - right_point_undistorted

        # Check that Y pixels are within tolerances
        if abs(disp[1]) > vertical_tolerance_px:
            raise ValueError(
                f"Pixels are not vertically aligned within tolerance: {disp[1]}"
            )
        return disp[0]

    def stereo_points_to_xyz(
        self, left_point: Union[NDArray, List], right_point: Union[NDArray, List]
    ) -> NDArray:
        """
        Given a left and right stereo point pair, return the real-world XYZ co-ordinates from the perspective of the left camera.

        *Note*: Input must be in (u,v) pixel form and non-rectified!

        This is mainly a helper function. For more detail, refer to the chained individual called functions.
        """
        disp = self.disparity_from_dual_points(left_point, right_point)
        left_rect_point = self.undistort_image_points(left_point, left=True)
        left_rect_point = [*left_rect_point, disp]
        xyz = self.points_px_to_3d_world_space(left_rect_point)
        return np.squeeze(xyz)
