from pathlib import Path
from typing import Optional, Tuple, Any
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
from numpy.typing import NDArray
from torch.autograd import Variable
import cv2
import logging
from machine_vision_acquisition_python.calibration.distortion import Calibration
from machine_vision_acquisition_python.calibration.libcalib import read_calib_parameters
from pyntcloud import PyntCloud
import pandas as pd

log = logging.getLogger(__name__)


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

    def remap(self, left: cv2.Mat, right: cv2.Mat) -> Tuple[cv2.Mat, cv2.Mat]:
        return cv2.remap(
            left, self.map_left_1, self.map_left_2, cv2.INTER_LINEAR
        ), cv2.remap(right, self.map_right_1, self.map_right_2, cv2.INTER_LINEAR)

    def calculate_disparity(self, left: cv2.Mat, right: cv2.Mat):
        """From two non-remapped images, return a single disparity image"""
        raise NotImplementedError()


class StereoProcessorHSM(StereoProcessor):
    def __init__(
        self,
        calibration_left: Calibration,
        calibration_right: Calibration,
        min_disparity=0,
        max_disparity=256,
        rescale_factor=1.0
    ) -> None:
        super().__init__(calibration_left, calibration_right)
        hsm_path = Path(r"/home/user/workspace/cfnmxl/high-res-stereo").resolve()
        hsm_script = hsm_path / "calculate_disparity.py"
        hsm_model_path = hsm_path / "pretrained-models" / "middlebury-final-768px.tar"
        if not hsm_script.exists() or not hsm_model_path.exists():
            raise FileNotFoundError(
                f"Could not locate HSM {hsm_script} or pre-trained model"
            )
        self.rescale_factor = rescale_factor
        self.min_disp = min_disparity
        self.max_disp = max_disparity
        self.num_disparities = max_disparity - min_disparity
        if min_disparity < 0:
            raise NotImplementedError("For now min_disparity must be > 0")
        if self.num_disparities % 16 != 0:
            # Todo: does it?
            raise ValueError(
                f"min_disparity and max_disparity must give a multiple of 16! got: {self.num_disparities}"
            )
        try:
            import torch
            import torchvision
            if not torch.cuda.is_available():
                raise NotImplementedError(
                    "Cannot run without CUDA (technically it can, but not implemented"
                )
        except ImportError as _:
            log.exception(
                "Cannot use StereoProcessorHSM without optional 'torch' installed. Run 'pip install torch'."
            )
        # patch in module
        import sys

        sys.path.append(str(hsm_path))

        from utils.model import load_model

        self.model, _, _ = load_model(
            model_path=str(hsm_model_path),
            max_disp=self.num_disparities,
            clean=-1,  # clean up output using entropy estimation
            cuda=True,
            data_parallel_model=False,
        )
        self.model.eval()
        self.module = self.model.module

    def calculate_disparity(self, left: cv2.Mat, right: cv2.Mat):
        from utils.inference import prepare_image_pair, perform_inference
        import torch
        imgL, imgR, img_size_in, img_size_in_scaled, img_size_net_in = prepare_image_pair(left, right, self.rescale_factor)
        # Load to GPU
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        pred_disp, entropy, _ = perform_inference(self.model, imgL, imgR, True)
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
        entropy = torch.squeeze(entropy).data.cpu().numpy()
        top_pad   = img_size_net_in[0]-img_size_in_scaled[0]
        left_pad  = img_size_net_in[1]-img_size_in_scaled[1]
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]
        entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad]
        # resize to highres
        pred_disp = cv2.resize(pred_disp/self.rescale_factor,(img_size_in[1],img_size_in[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = np.inf

        torch.cuda.empty_cache()

        disp_vis = (pred_disp/pred_disp[~invalid].max()*255).astype(np.uint8)
        ent_vis = entropy/entropy.max()*255
        return disp_vis

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

    def calculate_disparity(self, left: cv2.Mat, right: cv2.Mat):
        left_remapped, right_remapped = self.remap(left, right)  # Rectify
        left_remapped_gray = cv2.cvtColor(left_remapped, cv2.COLOR_BGR2GRAY)
        right_remapped_gray = cv2.cvtColor(right_remapped, cv2.COLOR_BGR2GRAY)
        disparity: cv2.Mat = self.bm.compute(left_remapped_gray, right_remapped_gray)
        disparity_8b = disparity.copy()
        disparity_8b = cv2.normalize(
            disparity, None, 0, np.iinfo(np.uint8).max, dtype=np.iinfo(np.uint8).max + 1
        )
        # Todo: get this working
        return disparity_8b


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

    # Manu's testing
    # stereo_rect = StereoProcessorOpenCVBM(
    #     calibrations[1], calibrations[2], min_disparity=500, max_disparity=804
    # )

    stereo_engine = StereoProcessorHSM(
        calibrations[1], calibrations[2], min_disparity=0, max_disparity=1600, rescale_factor=0.5
    )

    sourceLeft = cv2.imread(
        "tmp/testset/Lucid_213500023/img00000_exp1000_2022-04-29_16-24-09-963.tonemapped.png"
    )
    sourceRight = cv2.imread(
        "tmp/testset/Lucid_213500031/img00000_exp1000_2022-04-29_16-24-10-294.tonemapped.png"
    )

    output = stereo_engine.calculate_disparity(sourceLeft, sourceRight)

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
