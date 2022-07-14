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
    t=np.matmul(np.linalg.inv(t1), t2)
    #R=cv2.Rodrigues(t[0:3, 0:3])
    #R=R[0]
    #R=t[0:3, 0:3]
    #T=t[0:3, 3]

    #https://answers.opencv.org/question/89968/how-to-derive-relative-r-and-t-from-camera-extrinsics/
    R=np.matmul(np.linalg.inv(r1[0]), r2[0])
    T=np.matmul(r1[0].T, calibrations[2].tvec.T) - np.matmul(r1[0].T, calibrations[1].tvec.T)
    imageSize=[1920, 1080]
    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,4))
    P2 = np.zeros(shape=(3,4))
    Q = np.zeros(shape=(4,4))

   
    cv2.stereoRectify(calibrations[1].cameraMatrix, calibrations[1].distCoeffs, calibrations[2].cameraMatrix, calibrations[2].distCoeffs, imageSize, cv2.Rodrigues(R)[0], T, R1, R2, P1, P2, Q, cv2.CALIB_ZERO_DISPARITY, -1)

    mapLeftX, mapLeftY = cv2.initUndistortRectifyMap(calibrations[1].cameraMatrix, calibrations[1].distCoeffs, R1, P1,imageSize, cv2.CV_16SC2)
    mapRightX, mapRightY = cv2.initUndistortRectifyMap(calibrations[2].cameraMatrix, calibrations[2].distCoeffs, R2, P2, imageSize, cv2.CV_16SC2)

    #sourceLeft = cv2.imread('/home/user/workspace/cfnmxl/ssis_orchard_imaging/tmp/Lucid_213500023-20220708-102603-tonemapped/img00013_exp1000_2022-07-07_14-50-59-935.tonemapped.png')
    #sourceRight = cv2.imread('/home/user/workspace/cfnmxl/ssis_orchard_imaging/tmp/Lucid_213500031-20220708-102957-tonemapped/img00013_exp1000_2022-07-07_14-50-59-717.tonemapped.png')
    sourceLeft = cv2.imread('/home/user/workspace/cfnmxl/stereo-tuner/Lucid_213500023-img00000_exp1000_2022-04-29_16-35-55-769.tonemapped.png')
    sourceRight = cv2.imread('/home/user/workspace/cfnmxl/stereo-tuner/Lucid_213500031-img00000_exp1000_2022-04-29_16-35-55-468.tonemapped.png')
    cv2.imwrite('./tmp/sourceLeft.png', sourceLeft)
    cv2.imwrite('./tmp/sourceRight.png', sourceRight)

    # stereo blockmatcher works in single channel image
    sourceLeft = cv2.cvtColor(sourceLeft, cv2.COLOR_BGR2GRAY)
    sourceRight = cv2.cvtColor(sourceRight, cv2.COLOR_BGR2GRAY)
    leftImage = cv2.remap(sourceLeft, mapLeftX, mapLeftY, cv2.INTER_LINEAR)
    rightImage = cv2.remap(sourceRight, mapRightX, mapRightY, cv2.INTER_LINEAR)

    
    cv2.imwrite('./tmp/leftRect.png', leftImage)
    cv2.imwrite('./tmp/rightRect.png', rightImage)    

    stereo = cv2.StereoBM_create(640, 17)
    stereo.setPreFilterCap(7)
    stereo.setPreFilterSize(17)
    stereo.setMinDisparity(0)
    disparity = stereo.compute(leftImage, rightImage).astype(float) / 16.
    
    disp8bit=cv2.convertScaleAbs(disparity)
    cv2.imwrite('./tmp/disp.png', disp8bit)
 
    xyz=cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)

    np.savetxt('./tmp/ptcloud.xyz', np.reshape(xyz, (1080*1920,3)), delimiter=",", fmt='%.4f')