from pathlib import Path
from typing import Optional, Tuple, Any
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
from numpy.typing import NDArray
import cv2
import os
import logging
from machine_vision_acquisition_python.calibration.shared import Calibration
from machine_vision_acquisition_python.process.stereo.shared import StereoProcessor, StereoParams
import pandas as pd

log = logging.getLogger(__name__)
try:
    from pyntcloud import PyntCloud
except ImportError as _:
    log.warning("Failed to import pyntcloud, some functions may fail")
try:
    import torch
    import high_res_stereo.utils.model
    import high_res_stereo.utils.inference
    from torch.autograd import Variable
except ImportError as _:
    log.warning(
        "Failed to import torch and/or high_res_stereo, HSM functions may fail"
    )
    _HAS_HSM = False

class StereoProcessorHSM(StereoProcessor):
    def __init__(
        self,
        calibration_left: Calibration,
        calibration_right: Calibration,
        min_disparity=0,
        max_disparity=256,
        rescale_factor=1.0,
        clean=-1,
        hsm_model_path: Optional[Path] = None,
    ) -> None:
        super().__init__(calibration_left, calibration_right)
        hsm_model_path_env = os.environ.get("HSM_MODEL_PATH", None)
        if (
            hsm_model_path_env is not None
            and Path(hsm_model_path_env).resolve().exists()
        ):
            hsm_model_path = Path(hsm_model_path_env).resolve()
            log.debug(f"StereoProcessorHSM using model: {hsm_model_path}")
        if hsm_model_path is None:
            raise ValueError(
                "StereoProcessorHSM must be provided with a trained model either via 'hsm_model_path' or ENV 'HSM_MODEL_PATH'"
            )
        if not hsm_model_path.exists():
            raise FileNotFoundError(
                f"Could not locate HSM pre-trained model: {{hsm_model_path}}"
            )
        self.rescale_factor = rescale_factor
        self.min_disp = min_disparity
        self.max_disp = max_disparity
        self.num_disparities = max_disparity  # Note: HSM currently can't use a min disparity, clip it later on
        # if min_disparity != 0:
        #     raise NotImplementedError("For now min_disparity must be == 0")
        if min_disparity < 0:
            raise NotImplementedError("For now min_disparity must be > 0")
        if self.num_disparities % 16 != 0:
            # Todo: does it?
            raise ValueError(
                f"min_disparity and max_disparity must give a multiple of 16! got: {self.num_disparities}"
            )
        if clean != -1:
            log.warning(
                f"Values for clean of anything other than -1 are not understood"
            )
        self.clean = clean

        self.model, _, _ = high_res_stereo.utils.model.load_model(
            model_path=str(hsm_model_path),
            max_disp=self.num_disparities,
            # Todo: play around with clean parameter
            clean=-1,  # clean up output using entropy estimation, unsure of effect
            cuda=True,
            data_parallel_model=True,  # Setting this false seems to create terrible results
        )
        self.model.eval()
        self.module = self.model.module

    def calculate_disparity(self, left_remapped: cv2.Mat, right_remapped: cv2.Mat):
        (
            imgL,
            imgR,
            img_size_in,
            img_size_in_scaled,
            img_size_net_in,
        ) = high_res_stereo.utils.inference.prepare_image_pair(
            left_remapped, right_remapped, self.rescale_factor
        )
        # Load to GPU
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        disparity, entropy, _ = high_res_stereo.utils.inference.perform_inference(
            self.model, imgL, imgR, True
        )
        disparity = torch.squeeze(disparity).data.cpu().numpy()
        entropy = torch.squeeze(entropy).data.cpu().numpy()
        torch.cuda.empty_cache()
        top_pad = img_size_net_in[0] - img_size_in_scaled[0]
        left_pad = img_size_net_in[1] - img_size_in_scaled[1]
        disparity = disparity[top_pad:, : disparity.shape[1] - left_pad]
        entropy = entropy[top_pad:, : disparity.shape[1] - left_pad]
        # resize to highres
        if self.rescale_factor != 1.0:
            disparity = cv2.resize(
                disparity / self.rescale_factor,
                (img_size_in[1], img_size_in[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # clip to min/max disp values requested
        disparity = cv2.threshold(
            disparity, thresh=self.min_disp, maxval=0, type=cv2.THRESH_TOZERO
        )[1]
        disparity = cv2.threshold(
            disparity, thresh=self.max_disp, maxval=0, type=cv2.THRESH_TOZERO_INV
        )[1]
        # clip while keep inf
        invalid = np.logical_or(disparity == np.inf, disparity != disparity)
        disparity[invalid] = np.inf

        # clip to ROI
        disparity = self.apply_roi_to_disparity(disparity)
        return disparity

