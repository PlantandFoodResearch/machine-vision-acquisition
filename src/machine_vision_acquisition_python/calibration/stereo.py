from typing import Optional, Tuple
import numpy as np
import cv2
from machine_vision_acquisition_python.calibration.distortion import Calibration

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
    test = BlockMatcher(None)
