from pathlib import Path
from typing import Optional, Tuple
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
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
    
    #https://answers.opencv.org/question/89968/how-to-derive-relative-r-and-t-from-camera-extrinsics/
    # convert rotation vectors for each camera to 3x3 rotation matrices
    r1=cv2.Rodrigues(calibrations[1].rvec)
    r2=cv2.Rodrigues(calibrations[2].rvec)
        
    R=np.matmul(np.linalg.inv(r1[0]), r2[0])
    T=np.matmul(r1[0].T, calibrations[2].tvec.T) - np.matmul(r1[0].T, calibrations[1].tvec.T)
    
    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,4))
    P2 = np.zeros(shape=(3,4))
    Q = np.zeros(shape=(4,4))
    
    sourceLeft = cv2.imread('tmp/testset/Lucid_213500023/img00000_exp1000_2022-04-29_16-24-09-963.tonemapped.png')
    sourceRight = cv2.imread('tmp/testset/Lucid_213500031/img00000_exp1000_2022-04-29_16-24-10-294.tonemapped.png')
    imageSize=[sourceLeft.shape[1], sourceLeft.shape[0]]
   
    cv2.stereoRectify(calibrations[1].cameraMatrix, calibrations[1].distCoeffs, calibrations[2].cameraMatrix, calibrations[2].distCoeffs, imageSize, cv2.Rodrigues(R)[0], T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1)

    mapLeftX, mapLeftY = cv2.initUndistortRectifyMap(calibrations[1].cameraMatrix, calibrations[1].distCoeffs, R1, P1,imageSize, cv2.CV_16SC2)
    mapRightX, mapRightY = cv2.initUndistortRectifyMap(calibrations[2].cameraMatrix, calibrations[2].distCoeffs, R2, P2, imageSize, cv2.CV_16SC2)

    
    cv2.imwrite('./tmp/sourceLeft.png', sourceLeft)
    cv2.imwrite('./tmp/sourceRight.png', sourceRight)

    # need these for HSM matching
    colourLeftRemapped = cv2.remap(sourceLeft, mapLeftX, mapLeftY, cv2.INTER_LINEAR)
    colourRightRemapped = cv2.remap(sourceRight, mapRightX, mapRightY, cv2.INTER_LINEAR)
    
    cv2.imwrite('./tmp/leftColourRect.png', colourLeftRemapped)
    cv2.imwrite('./tmp/rightColourRect.png', colourRightRemapped)

    # stereo blockmatcher works in single channel image
    leftImage = cv2.cvtColor(colourLeftRemapped, cv2.COLOR_BGR2GRAY)
    rightImage = cv2.remap(cv2.cvtColor(sourceRight, cv2.COLOR_BGR2GRAY), mapRightX, mapRightY, cv2.INTER_LINEAR)
    
    cv2.imwrite('./tmp/leftRect.png', leftImage)
    cv2.imwrite('./tmp/rightRect.png', rightImage)    

    minDisparity = -256 # use this when not using CALIB_ZERO_DISPARITY. Use minDisparity=500 when flag is set. Max might be >630 
    numDisparities = 192
        
    stereo = cv2.StereoBM_create(numDisparities, 21)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereo.setPreFilterCap(7)
    stereo.setPreFilterSize(21)
    stereo.setSpeckleRange(4)
    stereo.setSpeckleWindowSize(61)
    stereo.setMinDisparity(minDisparity)
    disparity = stereo.compute(leftImage, rightImage).astype(np.float32) / 16.
    
    
    disp8bit=disparity*(256/numDisparities)+(-minDisparity * 256 /numDisparities)
    
    cv2.imwrite('./tmp/disp.png', disp8bit)
 
    xyz=cv2.reprojectImageTo3D(disparity, Q, True)
   
    points3D = np.reshape(xyz, (imageSize[0]*imageSize[1],3))
    d=np.reshape(disparity, (imageSize[0]*imageSize[1]))
    points3D[points3D == float('+inf')]=0
    points3D[points3D == float('-inf')]=0
    colours = np.reshape(colourLeftRemapped, (imageSize[0]*imageSize[1],3)) 
    
    disparityMask = d!=minDisparity-1
    points3D=points3D[disparityMask]
    colours=colours[disparityMask]

    d = {'x': points3D[:,0],'y': points3D[:,1],'z': points3D[:,2], 'red' : colours[:,0], 'green' : colours[:,1], 'blue' : colours[:,2]}
    cloud = PyntCloud(pd.DataFrame(data=d))

    cloud.to_file("./tmp/output.ply")