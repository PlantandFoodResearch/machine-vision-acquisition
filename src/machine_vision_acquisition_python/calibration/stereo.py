from pathlib import Path
from typing import Optional, Tuple, Any
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
from numpy.typing import NDArray
import torch
import high_res_stereo.utils.model
import high_res_stereo.utils.inference
from torch.autograd import Variable
import cv2
import os
import logging
from machine_vision_acquisition_python.calibration.libcalib import read_calib_parameters, Calibration
import pandas as pd

log = logging.getLogger(__name__)

try:
    from pyntcloud import PyntCloud
except ImportError as _:
    log.warning("Could not import optional dependency pyntcloud, some functions might cause exceptions")

_DEBUG_OUTPUT_FILES = True
_DEBUG_OUTPUT_PATH = (Path.cwd() / "tmp" / "debug").resolve()
if _DEBUG_OUTPUT_FILES:
    _DEBUG_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)


class BlockMatcher:
    def __init__(
        self, calibration: Calibration, image_shape: Optional[Tuple[int, int]] = None
    ) -> None:
        self.calibration = calibration
        # do some other stuff

    def bm_match(self, tetet) -> cv2.Mat:
        raise NotImplementedError()


class StereoParams:
    """Captures variables relating to cv2.stereoRectify: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6"""

    R1: Optional[NDArray]
    R2: Optional[NDArray]
    P1: Optional[NDArray]
    P2: Optional[NDArray]
    Q: Optional[NDArray]
    validROI1: Optional[Tuple]
    validROI2: Optional[Tuple]

    def __init__(self, R1, R2, P1, P2, Q, validROI1, validROI2) -> None:
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
        ):
            raise ValueError("Camera calibration image sizes don't match")
        self.image_size = (
            self.calibration_left.image_width,
            self.calibration_left.image_height,
        )  # (width, height)
        self.init_stereo_params()

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


    def apply_roi_to_disparity(self, disparity: cv2.Mat) -> cv2.Mat:
        masked_image = np.zeros(disparity.shape, disparity.dtype)
        roi = self.params.validROI1
        if roi is None or len(roi) != 4:
            raise ValueError("Invalid")
        x, y, w, h = roi
        masked_image[y:y+h,x:x+w] = disparity[y:y+h,x:x+w]
        return masked_image


    def remap(self, left: cv2.Mat, right: cv2.Mat) -> Tuple[cv2.Mat, cv2.Mat]:
        return cv2.remap(
            left, self.map_left_1, self.map_left_2, cv2.INTER_LINEAR
        ), cv2.remap(right, self.map_right_1, self.map_right_2, cv2.INTER_LINEAR)

    def calculate_disparity(self, left_remapped: cv2.Mat, right_remapped: cv2.Mat):
        """From two remapped images, return a single disparity image"""
        raise NotImplementedError()

    @staticmethod
    def normalise_disparity_16b(disparity: cv2.Mat) -> cv2.Mat:
        """Given the processed disparity image (with invalid pixels set to np.inf), return a normalised 16b disparity map"""
        # Normalise
        invalid = np.logical_or(disparity == np.inf,disparity!=disparity)
        new_max_value = np.iinfo(np.uint16).max
        old_max = disparity[~invalid].max()
        if old_max <= 0:
            log.debug("clipping old max < 0 to 1")
            old_max = 1.0
        log.debug(f"normalising disparity max from {old_max} to {new_max_value}")
        disp_16b = (disparity/old_max*new_max_value).astype(np.uint16)  # type: ignore
        return disp_16b

    @staticmethod
    def normalise_disparity_8b(disparity: cv2.Mat) -> cv2.Mat:
        """Given the processed disparity image (with invalid pixels set to np.inf), return a normalised 8b disparity map"""
        invalid = np.logical_or(disparity == np.inf,disparity!=disparity)
        new_max_value = np.iinfo(np.uint8).max
        old_max = disparity[~invalid].max()
        if old_max <= 0:
            log.debug("clipping old max < 0 to 1")
            old_max = 1.0
        log.debug(f"normalising disparity max from {old_max} to {new_max_value}")
        disp_8b = (disparity/old_max*new_max_value).astype(np.uint8)  # type: ignore
        return disp_8b

    @staticmethod
    def shift_disp_down(disparity: cv2.Mat) -> cv2.Mat:
        """Compress disparities towards 0 by removing empty space between 0 and first non-zero value"""
        # Shift back to within 0-255 range (min - max disp needs to be < 254)
        if disparity.max() <= 0:
            # nothing to do here, raise error?
            return disparity
        non_zero_min = disparity[disparity!=0].min()
        disparity_false = disparity - (non_zero_min + 1)
        disparity_false[disparity_false < 0] = 0
        return disparity_false

    def disparity_to_pointcloud(self, disparity: cv2.Mat, left_remapped: cv2.Mat, min_disp: Optional[int] = None, max_disp: Optional[int] = None) -> "PyntCloud":
        """Convert raw disparity output to coloured pointcloud"""
        log.warning(f"This code is experimental at best!")
        if min_disp is not None and max_disp:
            mask = np.ma.masked_inside(disparity, min_disp, max_disp)
            disparity = mask.data
        xyz = cv2.reprojectImageTo3D(disparity, self.params.Q, True)
        points3D = np.reshape(xyz, (self.image_size[0] * self.image_size[1], 3))
        colours = np.reshape(left_remapped, (self.image_size[0] * self.image_size[1], 3))

        data = np.concatenate([points3D, colours], axis=1)  # Combines xyz and BGR (in that order)

        # Clip outputs to Z values between 0.2 and 2.0
        idx = np.logical_and(
            data[:,2]<2.0,
            data[:,2]>0.2
            )
        data = data[idx]  # Only keep indicies that matched logical_and
        # PyntCloud epxects a Pandas DF. Explicitly name columns
        data_pd = pd.DataFrame.from_records(data, columns=[
            "x",
            "y",
            "z",
            "blue",
            "green",
            "red"
        ])
        # the merging will have converted the colour channels to floats. Revert them to uchar
        data_pd = data_pd.astype(
            {
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
            "blue": np.uint8,
            "green": np.uint8,
            "red": np.uint8
            }
        )
        cloud = PyntCloud(data_pd)
        return cloud

class StereoProcessorHSM(StereoProcessor):
    def __init__(
        self,
        calibration_left: Calibration,
        calibration_right: Calibration,
        min_disparity=0,
        max_disparity=256,
        rescale_factor=1.0,
        clean=-1,
        hsm_model_path: Optional[Path] = None
    ) -> None:
        super().__init__(calibration_left, calibration_right)
        hsm_model_path_env = os.environ.get("HSM_MODEL_PATH", None)
        if hsm_model_path_env is not None and Path(hsm_model_path_env).resolve().exists():
            hsm_model_path = Path(hsm_model_path_env).resolve()
            log.debug(f"StereoProcessorHSM using model: {hsm_model_path}")
        if hsm_model_path is None:
            raise ValueError("StereoProcessorHSM must be provided with a trained model either via 'hsm_model_path' or ENV 'HSM_MODEL_PATH'")
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
            log.warning(f"Values for clean of anything other than -1 are not understood")
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
        imgL, imgR, img_size_in, img_size_in_scaled, img_size_net_in = high_res_stereo.utils.inference.prepare_image_pair(left_remapped, right_remapped, self.rescale_factor)
        # Load to GPU
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        disparity, entropy, _ = high_res_stereo.utils.inference.perform_inference(self.model, imgL, imgR, True)
        disparity = torch.squeeze(disparity).data.cpu().numpy()
        entropy = torch.squeeze(entropy).data.cpu().numpy()
        torch.cuda.empty_cache()
        top_pad   = img_size_net_in[0]-img_size_in_scaled[0]
        left_pad  = img_size_net_in[1]-img_size_in_scaled[1]
        disparity = disparity[top_pad:,:disparity.shape[1]-left_pad]
        entropy = entropy[top_pad:,:disparity.shape[1]-left_pad]
        # resize to highres
        if self.rescale_factor != 1.0:
            disparity = cv2.resize(disparity/self.rescale_factor,(img_size_in[1],img_size_in[0]),interpolation=cv2.INTER_LINEAR)

        # clip to min/max disp values requested
        disparity = cv2.threshold(disparity, thresh=self.min_disp, maxval=0,  type=cv2.THRESH_TOZERO)[1]
        disparity = cv2.threshold(disparity, thresh=self.max_disp, maxval=0, type=cv2.THRESH_TOZERO_INV)[1]
        # clip while keep inf
        invalid = np.logical_or(disparity == np.inf,disparity!=disparity)
        disparity[invalid] = np.inf

        # clip to ROI
        disparity = self.apply_roi_to_disparity(disparity)
        return disparity


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
        if _DEBUG_OUTPUT_FILES:
            cv2.imwrite(str(_DEBUG_OUTPUT_PATH / "left_remapped_gray.png"), left_remapped_gray)
            cv2.imwrite(str(_DEBUG_OUTPUT_PATH / "right_remapped_gray.png"), right_remapped_gray)
        disparity: cv2.Mat = self.bm.compute(left_remapped_gray, right_remapped_gray)
        invalid = np.logical_or(disparity == np.inf,disparity!=disparity)
        disparity[invalid] = np.inf
        # clip to ROI
        disparity = self.apply_roi_to_disparity(disparity)
        return disparity


# Testing only, remove me
if __name__ == "__main__":

    calibio_json_path = Path(
        "/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-04-29/calibration-images/caloutput.json"
    )
    input_path = Path(
        "/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-04-29/images"
    )

    calibrations = read_calib_parameters(calibio_json_path)
    # Try match serial to folder path
    for calibration in calibrations:
        if calibration.serial in str(input_path.resolve()):
            log.debug(
                f"matched path component to calibration camera serial {calibration.serial}"
            )
            break

    # stereo_rect = StereoProcessorOpenCVBM(
    #     calibrations[1], calibrations[2], min_disparity=500, max_disparity=804
    # )

    stereo_engine = StereoProcessorHSM(
        calibrations[1], calibrations[2], min_disparity=340, max_disparity=960, rescale_factor=1.0,
        # hsm_model_path=Path(r"/home/user/workspace/cfnmxl/high-res-stereo/pretrained-models/middlebury-final-768px.tar")
    )

    # "\\storage.powerplant.pfr.co.nz\input\projects\dhs\smartsensingandimaging\development\fops\2022-07-21\2\Lucid_213500023\img00015_exp1000_2022-07-21_16-17-37-220.png"
    sourceLeft = cv2.imread(str(Path(
        r"/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-07-21/2/Lucid_213500023/img00015_exp1000_2022-07-21_16-17-37-220.png"
    ).resolve()), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    sourceRight = cv2.imread(str(Path(
        r"/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-07-21/2/Lucid_213500031/img00015_exp1000_2022-07-21_16-17-38-830.png"
    ).resolve()), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # sourceLeft = cv2.imread(str(Path(
    #     r"/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/tmp/stereo/Lucid_213500023/img00014_exp1000_2022-07-21_16-17-31-265.png"
    # ).resolve()), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    # sourceRight = cv2.imread(str(Path(
    #     r"/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/tmp/stereo/Lucid_213500031/img00014_exp1000_2022-07-21_16-17-32-860.png"
    # ).resolve()), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    image_left = cv2.cvtColor(sourceLeft, cv2.COLOR_BayerRG2RGB)
    image_right = cv2.cvtColor(sourceRight, cv2.COLOR_BayerRG2RGB)
    from machine_vision_acquisition_python.process.processing import cvt_tonemap_image
    image_left = cvt_tonemap_image(image_left)
    image_right = cvt_tonemap_image(image_right)
    # sourceRight = cv2.imread(
    #     "tmp/testset/Lucid_213500031/img00000_exp1000_2022-04-29_16-24-10-294.tonemapped.png"
    # )
    # sourceRight = cv2.imread(
    #     "tmp/testset/Lucid_213500031/img00000_exp1000_2022-04-29_16-24-10-294.tonemapped.png"
    # )

    left_remapped, right_remapped = stereo_engine.remap(image_left, image_right)  # Rectify
    if _DEBUG_OUTPUT_FILES:
        cv2.imwrite(str(_DEBUG_OUTPUT_PATH / "left_remapped.png"), left_remapped)
        cv2.imwrite(str(_DEBUG_OUTPUT_PATH / "right_remapped.png"), right_remapped)

    #
    # TEMP CODE
    # left_remapped = cv2.normalize(left_remapped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # right_remapped = cv2.normalize(right_remapped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    disparity_raw = stereo_engine.calculate_disparity(left_remapped, right_remapped)

    tx = -340  # px
    ty = 0  # px
    translation_matrix = np.float32([[1,0,tx], [0,1,ty]])  # type: ignore
    num_rows, num_cols = disparity_raw.shape[:2]
    img_translation = cv2.warpAffine(disparity_raw, translation_matrix, (num_cols, num_rows))

    # disparity_shifted =  stereo_engine.shift_disp_down(disparity_raw)
    # Shift back to within 0-255 range (min - max disp needs to be < 254)
    # disparity_false = disparity_raw - 511
    # disparity_false[disparity_false < 0] = 0
    # disparity_false_8b = disparity_false.astype(np.uint8)
    # disparity_raw_thresh = cv2.threshold(disparity_raw, thresh=400, maxval=0,  type=cv2.THRESH_TOZERO)[1]
    # disparity_raw_thresh = cv2.threshold(disparity_raw_thresh, thresh=1000, maxval=0, type=cv2.THRESH_TOZERO_INV)[1]
    disparity_normalised = StereoProcessor.normalise_disparity_16b(img_translation)
    if _DEBUG_OUTPUT_FILES:
        cv2.imwrite(str(_DEBUG_OUTPUT_PATH / "disp-shifty.png"), disparity_normalised)

    # clipping_mask = np.logical_or(disparity_raw <= 500, disparity_raw >= 100)
    # xyz = stereo_engine.disparity_to_pointcloud(disparity_raw, left_remapped, 500, 800)
    # xyz.to_file(str(_DEBUG_OUTPUT_PATH / "disp-clipped-moved.ply"))


    raise NotImplementedError()
    # https://answers.opencv.org/question/89968/how-to-derive-relative-r-and-t-from-camera-extrinsics/
    # convert rotation vectors for each camera to 3x3 rotation matrices
    r1 = cv2.Rodrigues(calibrations[1].rvec)
    r2 = cv2.Rodrigues(calibrations[2].rvec)

    R = np.matmul(np.linalg.inv(r1[0]), r2[0])
    T = np.matmul(r1[0].T, calibrations[2].tvec.T) - np.matmul(
        r1[0].T, calibrations[1].tvec.T
    )

    R1 = np.zeros(shape=(3, 3))
    R2 = np.zeros(shape=(3, 3))
    P1 = np.zeros(shape=(3, 4))
    P2 = np.zeros(shape=(3, 4))
    Q = np.zeros(shape=(4, 4))

    sourceLeft = cv2.imread(
        "tmp/testset/Lucid_213500023/img00000_exp1000_2022-04-29_16-24-09-963.tonemapped.png"
    )
    sourceRight = cv2.imread(
        "tmp/testset/Lucid_213500031/img00000_exp1000_2022-04-29_16-24-10-294.tonemapped.png"
    )
    imageSize = [sourceLeft.shape[1], sourceLeft.shape[0]]

    cv2.stereoRectify(
        calibrations[1].cameraMatrix,
        calibrations[1].distCoeffs,
        calibrations[2].cameraMatrix,
        calibrations[2].distCoeffs,
        imageSize,
        cv2.Rodrigues(R)[0],
        T,
        R1,
        R2,
        P1,
        P2,
        Q,
        CALIB_ZERO_DISPARITY,
        -1,
    )

    mapLeftX, mapLeftY = cv2.initUndistortRectifyMap(
        calibrations[1].cameraMatrix,
        calibrations[1].distCoeffs,
        R1,
        P1,
        imageSize,
        cv2.CV_16SC2,
    )
    mapRightX, mapRightY = cv2.initUndistortRectifyMap(
        calibrations[2].cameraMatrix,
        calibrations[2].distCoeffs,
        R2,
        P2,
        imageSize,
        cv2.CV_16SC2,
    )

    cv2.imwrite("./tmp/sourceLeft.png", sourceLeft)
    cv2.imwrite("./tmp/sourceRight.png", sourceRight)

    # need these for HSM matching
    colourLeftRemapped = cv2.remap(sourceLeft, mapLeftX, mapLeftY, cv2.INTER_LINEAR)
    colourRightRemapped = cv2.remap(sourceRight, mapRightX, mapRightY, cv2.INTER_LINEAR)

    cv2.imwrite("./tmp/leftColourRect.png", colourLeftRemapped)
    cv2.imwrite("./tmp/rightColourRect.png", colourRightRemapped)

    # stereo blockmatcher works in single channel image
    leftImage = cv2.cvtColor(colourLeftRemapped, cv2.COLOR_BGR2GRAY)
    rightImage = cv2.remap(
        cv2.cvtColor(sourceRight, cv2.COLOR_BGR2GRAY),
        mapRightX,
        mapRightY,
        cv2.INTER_LINEAR,
    )

    cv2.imwrite("./tmp/leftRect.png", leftImage)
    cv2.imwrite("./tmp/rightRect.png", rightImage)

    minDisparity = (
        -256
    )  # use this when not using CALIB_ZERO_DISPARITY. Use minDisparity=500 when flag is set. Max might be >630
    numDisparities = 192

    stereo = cv2.StereoBM_create(numDisparities, 21)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereo.setPreFilterCap(7)
    stereo.setPreFilterSize(21)
    stereo.setSpeckleRange(4)
    stereo.setSpeckleWindowSize(61)
    stereo.setMinDisparity(minDisparity)
    disparity = stereo.compute(leftImage, rightImage).astype(np.float32) / 16.0

    disp8bit = disparity * (256 / numDisparities) + (
        -minDisparity * 256 / numDisparities
    )

    cv2.imwrite("./tmp/disp.png", disp8bit)

    xyz = cv2.reprojectImageTo3D(disparity, Q, True)

    points3D = np.reshape(xyz, (imageSize[0] * imageSize[1], 3))
    d = np.reshape(disparity, (imageSize[0] * imageSize[1]))
    points3D[points3D == float("+inf")] = 0
    points3D[points3D == float("-inf")] = 0
    colours = np.reshape(colourLeftRemapped, (imageSize[0] * imageSize[1], 3))

    disparityMask = d != minDisparity - 1
    points3D = points3D[disparityMask]
    colours = colours[disparityMask]

    d = {
        "x": points3D[:, 0],
        "y": points3D[:, 1],
        "z": points3D[:, 2],
        "red": colours[:, 0],
        "green": colours[:, 1],
        "blue": colours[:, 2],
    }
    cloud = PyntCloud(pd.DataFrame(data=d))

    cloud.to_file("./tmp/output.ply")
