# This file contains stereo processing code
from typing import Optional, Tuple, Union
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
from numpy.typing import NDArray
import cv2
import cv2.fisheye
import logging
from machine_vision_acquisition_python.calibration.shared import Calibration, CameraModel
log = logging.getLogger(__name__)
try:
    from pyntcloud import PyntCloud
except ImportError as _:
    log.warning("Failed to import pyntcloud, some functions may fail")



class StereoParams:
    """Captures variables relating to cv2.stereoRectify: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6"""

    R1: Optional[NDArray]
    R2: Optional[NDArray]
    P1: Optional[NDArray]
    P2: Optional[NDArray]
    Q: Optional[NDArray]
    validROI1: Optional[Tuple]
    validROI2: Optional[Tuple]

    def __init__(self, R1, R2, P1, P2, Q, validROI1=None, validROI2=None) -> None:
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.validROI1 = validROI1
        self.validROI2 = validROI2


class StereoProcessor:
    def __init__(
        self,
        calibration_left: Calibration,
        calibration_right: Calibration,
    ) -> None:
        self.calibration_left = calibration_left
        self.calibration_right = calibration_right
        if (
            calibration_left.image_width != calibration_right.image_width
            or calibration_left.image_height != calibration_right.image_height
            or self.calibration_left.camera_model != self.calibration_right.camera_model
        ):
            raise ValueError("Camera calibration image sizes or models don't match")
        self.image_size = (
            self.calibration_left.image_width,
            self.calibration_left.image_height,
        )  # (width, height)
        self.camera_model: CameraModel = self.calibration_left.camera_model
        self.init_stereo_params()

    def init_opencv_model_params(self):
        # Generate stereo params
        self.params = StereoParams(
            *cv2.stereoRectify(
                self.calibration_left.cameraMatrix,
                self.calibration_left.distCoeffs,
                self.calibration_right.cameraMatrix,
                self.calibration_right.distCoeffs,
                self.image_size,
                cv2.Rodrigues(self.R)[0],
                self.T,
                flags=CALIB_ZERO_DISPARITY,
                alpha=-1,
            )
        )

        self.map_left_1, self.map_left_2 = cv2.initUndistortRectifyMap(
            self.calibration_left.cameraMatrix,
            self.calibration_left.distCoeffs,
            self.params.R1,
            self.params.P1,
            self.image_size,
            cv2.CV_16SC2,
        )
        self.map_right_1, self.map_right_2 = cv2.initUndistortRectifyMap(
            self.calibration_right.cameraMatrix,
            self.calibration_right.distCoeffs,
            self.params.R2,
            self.params.P2,
            self.image_size,
            cv2.CV_16SC2,
        )

    def init_opencvfisheye_model_params(self):
        self.params = StereoParams(
            *cv2.fisheye.stereoRectify(
                self.calibration_left.cameraMatrix,
                self.calibration_left.distCoeffs,
                self.calibration_right.cameraMatrix,
                self.calibration_right.distCoeffs,
                self.image_size,
                cv2.Rodrigues(self.R)[0],
                self.T,
                flags=CALIB_ZERO_DISPARITY,
            )
        )

        self.map_left_1, self.map_left_2 = cv2.fisheye.initUndistortRectifyMap(
            self.calibration_left.cameraMatrix,
            self.calibration_left.distCoeffs,
            self.params.R1,
            self.params.P1,
            self.image_size,
            cv2.CV_16SC2,
        )
        self.map_right_1, self.map_right_2 = cv2.fisheye.initUndistortRectifyMap(
            self.calibration_right.cameraMatrix,
            self.calibration_right.distCoeffs,
            self.params.R2,
            self.params.P2,
            self.image_size,
            cv2.CV_16SC2,
        )

    def init_stereo_params(self):
        # https://answers.opencv.org/question/89968/how-to-derive-relative-r-and-t-from-camera-extrinsics/
        # convert rotation vectors for each camera to 3x3 rotation matrices
        self.r1 = cv2.Rodrigues(self.calibration_left.rvec)
        self.r2 = cv2.Rodrigues(self.calibration_right.rvec)

        # Ensure that r1 and r2 are relative to each other
        self.R = np.matmul(np.linalg.inv(self.r1[0]), self.r2[0])
        self.T = np.matmul(self.r1[0].T, self.calibration_right.tvec.T) - np.matmul(
            self.r1[0].T, self.calibration_left.tvec.T
        )

        if self.camera_model == CameraModel.OpenCV:
            self.init_opencv_model_params()
        elif self.camera_model == CameraModel.OpenCVFisheye:
            self.init_opencvfisheye_model_params()
        else:
            raise NotImplementedError(f"Camera model {self.camera_model} not supported (yet!)")

    def apply_roi_to_disparity(self, disparity: cv2.Mat) -> cv2.Mat:
        masked_image = np.zeros(disparity.shape, disparity.dtype)
        roi = self.params.validROI1
        if roi is None or len(roi) != 4:
            raise ValueError("Invalid")
        x, y, w, h = roi
        masked_image[y : y + h, x : x + w] = disparity[y : y + h, x : x + w]
        return masked_image

    def remap(self, left: cv2.Mat, right: cv2.Mat) -> Tuple[cv2.Mat, cv2.Mat]:
        return cv2.remap(
            left, self.map_left_1, self.map_left_2, cv2.INTER_LINEAR
        ), cv2.remap(right, self.map_right_1, self.map_right_2, cv2.INTER_LINEAR)

    def calculate_disparity(self, left_remapped: cv2.Mat, right_remapped: cv2.Mat):
        """From two remapped images, return a single disparity image"""
        raise NotImplementedError()

    @property
    def baseline_mm(self):
        """Returns the stereo camera baseline in mm units (the norm of the t_vec)"""
        return np.linalg.norm(self.T)

    @staticmethod
    def normalise_disparity_16b(disparity: cv2.Mat) -> cv2.Mat:
        """Given the processed disparity image (with invalid pixels set to np.inf), return a normalised 16b disparity map"""
        # Normalise
        invalid = np.logical_or(disparity == np.inf, disparity != disparity)
        new_max_value = np.iinfo(np.uint16).max
        old_max = disparity[~invalid].max()
        if old_max <= 0:
            log.debug("clipping old max < 0 to 1")
            old_max = 1.0
        log.debug(f"normalising disparity max from {old_max} to {new_max_value}")
        disp_16b = (disparity / old_max * new_max_value).astype(np.uint16)  # type: ignore
        return disp_16b

    @staticmethod
    def normalise_disparity_8b(disparity: cv2.Mat) -> cv2.Mat:
        """Given the processed disparity image (with invalid pixels set to np.inf), return a normalised 8b disparity map"""
        invalid = np.logical_or(disparity == np.inf, disparity != disparity)
        new_max_value = np.iinfo(np.uint8).max
        old_max = disparity[~invalid].max()
        if old_max <= 0:
            log.debug("clipping old max < 0 to 1")
            old_max = 1.0
        log.debug(f"normalising disparity max from {old_max} to {new_max_value}")
        disp_8b = (disparity / old_max * new_max_value).astype(np.uint8)  # type: ignore
        return disp_8b

    @staticmethod
    def shift_disp_down(disparity: cv2.Mat) -> cv2.Mat:
        """Compress disparities towards 0 by removing empty space between 0 and first non-zero value"""
        # Shift back to within 0-255 range (min - max disp needs to be < 254)
        if disparity.max() <= 0:
            # nothing to do here, raise error?
            return disparity
        non_zero_min = disparity[disparity != 0].min()
        disparity_false = disparity - (non_zero_min + 1)
        disparity_false[disparity_false < 0] = 0
        return disparity_false

    def disparity_to_depth_mm(self, disparity_value: Union[int, float]) -> float:
        """
        Uses knowledge of camera calibration to project left hand disparity to depth in mm.
        Source: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
        """
        focallength_px = self.calibration_left.cameraMatrix[0, 0]
        # doffs = self.calibration_right.cameraMatrix[0, 2] - self.calibration_left.cameraMatrix[0, 2]
        doffs = 0  # Because we use stereoRectify with the flag CALIB_ZERO_DISPARITY, image centers should be aligned
        baseline_mm = np.linalg.norm(self.T)
        return baseline_mm * focallength_px / (disparity_value + doffs)

    # def disparity_to_pointcloud(
    #     self,
    #     disparity: cv2.Mat,
    #     left_remapped: cv2.Mat,
    #     min_disp: Optional[int] = None,
    #     max_disp: Optional[int] = None,
    # ) -> "PyntCloud":
    #     """Convert raw disparity output to coloured pointcloud"""
    #     log.warning(f"This code is experimental at best!")
    #     if min_disp is not None and max_disp:
    #         mask = np.ma.masked_inside(disparity, min_disp, max_disp)
    #         disparity = mask.data
    #     xyz = cv2.reprojectImageTo3D(disparity, self.params.Q, True)
    #     points3D = np.reshape(xyz, (self.image_size[0] * self.image_size[1], 3))
    #     colours = np.reshape(
    #         left_remapped, (self.image_size[0] * self.image_size[1], 3)
    #     )

    #     data = np.concatenate(
    #         [points3D, colours], axis=1
    #     )  # Combines xyz and BGR (in that order)

    #     # Clip outputs to Z values between 0.2 and 2.0
    #     idx = np.logical_and(data[:, 2] < 2.0, data[:, 2] > 0.2)
    #     data = data[idx]  # Only keep indicies that matched logical_and
    #     # PyntCloud epxects a Pandas DF. Explicitly name columns
    #     data_pd = pd.DataFrame.from_records(
    #         data, columns=["x", "y", "z", "blue", "green", "red"]
    #     )
    #     # the merging will have converted the colour channels to floats. Revert them to uchar
    #     data_pd = data_pd.astype(
    #         {
    #             "x": np.float32,
    #             "y": np.float32,
    #             "z": np.float32,
    #             "blue": np.uint8,
    #             "green": np.uint8,
    #             "red": np.uint8,
    #         }
    #     )
    #     cloud = PyntCloud(data_pd)
    #     return cloud
