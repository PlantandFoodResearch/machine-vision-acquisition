from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
import logging
from machine_vision_acquisition_python.calibration.distortion import Calibration
from machine_vision_acquisition_python.calibration.libcalib import read_calib_parameters

log = logging.getLogger(__name__)

class BlockMatcher:
    def __init__(
        self, calibration: Calibration, image_shape: Optional[Tuple[int, int]] = None
    ) -> None:
        self.calibration = calibration
        # do some other stuff


    def bm_match(self, tetet) -> cv2.Mat:
        raise NotImplementedError()



# Testing only, remove me
if __name__ == "__main__":
    
    calibio_json_path = Path('/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-04-29/calibration-images/caloutput.json')
    input_path = Path('/mnt/powerplant/input/projects/dhs/smartsensingandimaging/development/fops/2022-04-29/images')

    calibrations = read_calib_parameters(calibio_json_path)
    # Try match serial to folder path
    for calibration in calibrations:
        if calibration.serial in str(input_path.resolve()):
            log.debug(
                f"matched path component to calibration camera serial {calibration.serial}"
            )
            break
    
    #build 4x4 homogeneous coordinates for each camera
    t1=np.eye(4, dtype=float)
    np.put(t1, [3, 7, 11], calibrations[1].tvec)
    r1=cv2.Rodrigues(calibrations[1].rvec)
    np.put(t1, [0,1,2,4,5,6,8,9,10], r1[0])

    t2=np.eye(4, dtype=float)
    np.put(t2, [3, 7, 11], calibrations[2].tvec)
    r2=cv2.Rodrigues(calibrations[2].rvec)
    np.put(t2, [0,1,2,4,5,6,8,9,10], r2[0])

    # calculate cam2 pose realtive to cam1
    t=np.matmul(t2, np.linalg.inv(t1))
    R=cv2.Rodrigues(t[0:3, 0:3])
    R=R[0]
    T=t[0:3, 3]

    imageSize=[1920, 1080]
    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,3))
    P2 = np.zeros(shape=(3,3))
    cv2.stereoRectify(calibrations[1].cameraMatrix, calibrations[1].distCoeffs, calibrations[2].cameraMatrix, calibrations[2].distCoeffs, imageSize, R, T, R1, R2, P1, P2, flags=cv2.CALIB_ZERO_DISPARITY)

    mapLeftX, mapLeftY = cv2.initUndistortRectifyMap(calibrations[1].cameraMatrix, calibrations[1].distCoeffs, R1, calibrations[1].cameraMatrix,imageSize, cv2.CV_32FC1)
    mapRightX, mapRightY = cv2.initUndistortRectifyMap(calibrations[2].cameraMatrix, calibrations[2].distCoeffs, R2, calibrations[2].cameraMatrix, imageSize, cv2.CV_32FC1)

    leftImage = cv2.remap(sourceLeft, mapLeftY, mapLeftY)
    rightImage = cv2.remap(sourceRight, mapRightX, mapRightY)

    blockMatcher = cv2.StereoBM_create(32, 16)


    test = BlockMatcher(None)