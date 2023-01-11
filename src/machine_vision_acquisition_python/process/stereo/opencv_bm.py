from pathlib import Path
from typing import Optional, Tuple, Any
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
from numpy.typing import NDArray
import cv2
import logging
from machine_vision_acquisition_python.calibration.shared import Calibration
from machine_vision_acquisition_python.process.stereo.shared import (
    StereoProcessor,
    StereoParams,
)
import pandas as pd

log = logging.getLogger(__name__)


class StereoProcessorOpenCVBM(StereoProcessor):
    def __init__(
        self,
        calibration_left: Calibration,
        calibration_right: Calibration,
        min_disparity=0,
        max_disparity=256,
    ) -> None:
        super().__init__(calibration_left, calibration_right)
        self.min_disp = min_disparity
        self.max_disp = max_disparity
        self.num_disparities = max_disparity - min_disparity
        if self.num_disparities % 16 != 0:
            raise ValueError(
                f"min_disparity and max_disparity must give a multiple of 16! got: {self.num_disparities}"
            )
        elif self.num_disparities < 16:
            raise ValueError(f"invalid min/max disparity values")
        self.bm: cv2.StereoBM = cv2.StereoBM_create(
            numDisparities=self.num_disparities, blockSize=21
        )
        self.bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
        self.bm.setPreFilterCap(7)
        self.bm.setPreFilterSize(21)
        self.bm.setSpeckleRange(4)
        self.bm.setSpeckleWindowSize(61)
        self.bm.setMinDisparity(min_disparity)

    def calculate_disparity(self, left_remapped: cv2.Mat, right_remapped: cv2.Mat):
        left_remapped_gray = cv2.cvtColor(left_remapped, cv2.COLOR_BGR2GRAY)
        right_remapped_gray = cv2.cvtColor(right_remapped, cv2.COLOR_BGR2GRAY)
        disparity: cv2.Mat = self.bm.compute(left_remapped_gray, right_remapped_gray)
        invalid = np.logical_or(disparity == np.inf, disparity != disparity)
        disparity[invalid] = 0 if disparity.dtype == np.int16 else np.inf
        # clip to ROI
        disparity = self.apply_roi_to_disparity(disparity)
        return disparity
